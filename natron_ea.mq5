#property copyright "Natron"
#property link      "https://natron.ai"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include <stdlib.mqh>

input string ServerURL = "http://127.0.0.1:8080/predict";
input int SequenceLength = 96;
input double BuyThreshold = 0.6;
input double SellThreshold = 0.6;
input double MaxRiskPerTrade = 0.01;

CTrade trade;
datetime last_request_time = 0;

string BuildPayload();
bool SendPredictionRequest(string payload, string &response);
void HandleResponse(const string &json_text);
void DrawDashboard(double buy_prob, double sell_prob, double direction_up, string regime, double confidence);

int OnInit()
{
   Comment("Natron EA initialised. Ensure ", ServerURL, " is authorised in Tools > Options > Expert Advisors.");
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_time == last_request_time)
      return;

   last_request_time = current_bar_time;

   string payload = BuildPayload();
   if(payload == "")
      return;

   string response;
   if(!SendPredictionRequest(payload, response))
   {
      Print("Natron EA: failed to obtain response");
      return;
   }
   HandleResponse(response);
}

string BuildPayload()
{
   MqlRates rates[];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, SequenceLength, rates) != SequenceLength)
   {
      Print("Natron EA: insufficient candles for payload");
      return "";
   }

   ArraySetAsSeries(rates, true);
   string candles = "[";
   for(int i = SequenceLength - 1; i >= 0; --i)
   {
      string time_iso = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS);
      candles += StringFormat("{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%.0f}", time_iso, rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume);
      if(i != 0)
         candles += ",";
   }
   candles += "]";
   string payload = StringFormat("{\"symbol\":\"%s\",\"candles\":%s}", _Symbol, candles);
   return payload;
}

bool SendPredictionRequest(string payload, string &response)
{
   char post[];
   StringToCharArray(payload, post);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   int timeout = 5000;
   int res = WebRequest("POST", ServerURL, headers, timeout, post, result, NULL);
   if(res != 200)
   {
      PrintFormat("Natron EA: WebRequest error %d", res);
      return false;
   }
   response = CharArrayToString(result);
   return true;
}

void HandleResponse(const string &json_text)
{
   CJAVal json;
   if(!json.Deserialize(json_text))
   {
      Print("Natron EA: failed to parse JSON response");
      return;
   }

   double buy_prob = json["buy_prob"].ToDouble();
   double sell_prob = json["sell_prob"].ToDouble();
   double direction_up = json["direction_up"].ToDouble();
   string regime = json["regime"].ToStr();
   double confidence = json["confidence"].ToDouble();

   DrawDashboard(buy_prob, sell_prob, direction_up, regime, confidence);

   double lot_size = NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY) * MaxRiskPerTrade / 100000.0, 2);

   if(buy_prob >= BuyThreshold && sell_prob < SellThreshold)
   {
      trade.Buy(lot_size, _Symbol);
   }
   else if(sell_prob >= SellThreshold && buy_prob < BuyThreshold)
   {
      trade.Sell(lot_size, _Symbol);
   }
}

void DrawDashboard(double buy_prob, double sell_prob, double direction_up, string regime, double confidence)
{
   string text = StringFormat("Natron\nBuy: %.2f%%\nSell: %.2f%%\nUp: %.2f%%\nRegime: %s\nConf: %.2f%%",
      buy_prob * 100.0, sell_prob * 100.0, direction_up * 100.0, regime, confidence * 100.0);
   Comment(text);
}
