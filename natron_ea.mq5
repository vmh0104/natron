//+------------------------------------------------------------------+
//| Natron Transformer Expert Advisor                               |
//| Connects to Natron Python server via TCP to fetch trading signal |
//+------------------------------------------------------------------+
#property copyright "Natron"
#property version   "2.0"
#property strict
#property description "Natron Transformer EA integrates with the Natron GPU model via TCP bridge."

#include <Trade\Trade.mqh>
#include <JSON\json.mqh>

input string InpServerHost   = "127.0.0.1";
input int    InpServerPort   = 9000;
input int    InpSequenceSize = 96;
input double InpBuyThreshold  = 0.6;
input double InpSellThreshold = 0.6;
input double InpLots          = 0.1;
input int    InpRefreshSec    = 60;
input bool   InpEnableTrading = true;

CTrade trade;
datetime last_request_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(InpRefreshSec);
   Comment("Natron EA initialized.");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   Comment("");
  }

//+------------------------------------------------------------------+
//| Timer event                                                      |
//+------------------------------------------------------------------+
void OnTimer()
  {
   datetime now = TimeCurrent();
   if(now == last_request_time)
      return;
   last_request_time = now;

   string payload;
   if(!BuildPayload(payload))
      return;

   string response;
   if(!SendRequest(payload, response))
     {
      Print("Natron EA: failed to receive response.");
      return;
     }

   ProcessResponse(response);
  }

//+------------------------------------------------------------------+
//| Build JSON payload with last N candles                           |
//+------------------------------------------------------------------+
bool BuildPayload(string &payload)
  {
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, InpSequenceSize, rates);
   if(copied < InpSequenceSize)
     {
      PrintFormat("Natron EA: insufficient candles (%d/%d).", copied, InpSequenceSize);
      return(false);
     }

   ArraySetAsSeries(rates, true);

   string candles = "";
   for(int i = InpSequenceSize - 1; i >= 0; --i)
     {
      string candle = StringFormat(
         "{\"time\":\"%s\",\"open\":%.6f,\"high\":%.6f,\"low\":%.6f,\"close\":%.6f,\"volume\":%.2f}",
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
         rates[i].open, rates[i].high, rates[i].low, rates[i].close, rates[i].tick_volume
      );
      if(i < InpSequenceSize - 1)
         candles += ",";
      candles += candle;
     }

   payload = StringFormat("{\"symbol\":\"%s\",\"candles\":[%s]}\n", _Symbol, candles);
   return(true);
  }

//+------------------------------------------------------------------+
//| Send payload via TCP socket                                      |
//+------------------------------------------------------------------+
bool SendRequest(const string payload, string &response)
  {
   int socket = SocketCreate();
   if(socket == INVALID_HANDLE)
     {
      Print("Natron EA: unable to create socket.");
      return(false);
     }

   if(!SocketConnect(socket, InpServerHost, (ushort)InpServerPort, 5000))
     {
      Print("Natron EA: connection failed.");
      SocketClose(socket);
      return(false);
     }

   char send_buffer[];
   StringToCharArray(payload, send_buffer);
   int to_send = ArraySize(send_buffer) - 1;
   int sent = SocketSend(socket, send_buffer, to_send);
   if(sent != to_send)
     {
      Print("Natron EA: failed to send full payload.");
      SocketClose(socket);
      return(false);
     }

   char recv_buffer[4096];
   int received = SocketRead(socket, recv_buffer, sizeof(recv_buffer) - 1, 5000);
   SocketClose(socket);

   if(received <= 0)
     {
      Print("Natron EA: empty response.");
      return(false);
     }

   recv_buffer[received] = '\0';
   response = CharArrayToString(recv_buffer, 0, received);
   return(true);
  }

//+------------------------------------------------------------------+
//| Process JSON response                                            |
//+------------------------------------------------------------------+
void ProcessResponse(const string response)
  {
   CJAVal json;
   if(!json.Deserialize(response))
     {
      Print("Natron EA: JSON parse error.");
      return;
     }

   double buy_prob = json["buy_prob"].ToDouble();
   double sell_prob = json["sell_prob"].ToDouble();
   double direction_up = json["direction_up"].ToDouble();
   string regime = json["regime"].ToStr();
   double confidence = json["confidence"].ToDouble();

   Comment(StringFormat("Natron\nBuy: %.2f  Sell: %.2f\nDirectionUp: %.2f\nRegime: %s\nConfidence: %.2f",
                        buy_prob, sell_prob, direction_up, regime, confidence));

   if(!InpEnableTrading)
      return;

   ManagePositions(buy_prob, sell_prob, regime, confidence);
  }

//+------------------------------------------------------------------+
//| Trading logic                                                    |
//+------------------------------------------------------------------+
void ManagePositions(double buy_prob, double sell_prob, string regime, double confidence)
  {
   int total = PositionsTotal();
   bool has_buy = PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
   bool has_sell = PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL;

   double sl=0.0, tp=0.0;
   double atr = iATR(_Symbol, PERIOD_CURRENT, 14, 0);
   if(atr > 0)
     {
      sl = atr * 1.5;
      tp = atr * 3.0;
     }

   if(buy_prob >= InpBuyThreshold && !has_buy)
     {
      if(has_sell)
         trade.PositionClose(_Symbol);
      trade.Buy(InpLots, _Symbol, 0.0, sl, tp, "Natron Buy");
     }
   else if(sell_prob >= InpSellThreshold && !has_sell)
     {
      if(has_buy)
         trade.PositionClose(_Symbol);
      trade.Sell(InpLots, _Symbol, 0.0, sl, tp, "Natron Sell");
     }
   else if(confidence < 0.4 && total > 0)
     {
      trade.PositionClose(_Symbol);
     }
  }
