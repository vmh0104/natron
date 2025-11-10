#property copyright "Natron"
#property link      "https://natron.ai"
#property version   "1.0"
#property strict

#include <Trade\Trade.mqh>
#include <WinAPI\winsock2.mqh>

input string InpHost            = "127.0.0.1";
input int    InpTcpPort         = 7600;
input int    InpHttpPort        = 8080;
input bool   InpUseHttp         = false;
input int    InpSequenceLength  = 96;
input double InpBuyThreshold    = 0.65;
input double InpSellThreshold   = 0.65;
input double InpLots            = 0.10;
input int    InpSlippage        = 5;
input int    InpMagic           = 42069;
input int    InpRequestInterval = 60; // seconds between server calls

CTrade      trade;
CWinSock    socketClient;
datetime    lastRequestTime = 0;
double      lastBuyProb = 0.0;
double      lastSellProb = 0.0;
double      lastDirectionUp = 0.0;
string      lastRegime = "";

//--- Helper prototypes
bool   EnsureConnection();
bool   SendRequest(string payload, string &response);
bool   SendHttpRequest(string payload, string &response);
bool   SendTcpRequest(string payload, string &response);
string BuildPayload();
double ExtractJsonDouble(const string json, const string key);
string ExtractJsonString(const string json, const string key);
void   UpdateChart();
void   HandleSignals();
bool   PositionsExist(int type);

int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   if(!InpUseHttp && !EnsureConnection())
     {
      Print("Natron EA failed to connect to socket server");
      return(INIT_FAILED);
     }
   EventSetTimer(1);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   socketClient.Close();
   Comment("");
  }

void OnTimer()
  {
   if(TimeCurrent() - lastRequestTime < InpRequestInterval)
      return;
   string payload = BuildPayload();
   if(payload == "")
      return;

   string response;
   if(!SendRequest(payload, response))
     {
      Print("Natron EA: request failed");
      return;
     }

   lastBuyProb = ExtractJsonDouble(response, "buy_prob");
   lastSellProb = ExtractJsonDouble(response, "sell_prob");
   lastDirectionUp = ExtractJsonDouble(response, "direction_up");
   lastRegime = ExtractJsonString(response, "regime");
   lastRequestTime = TimeCurrent();

   UpdateChart();
   HandleSignals();
  }

void OnTick()
  {
   // No-op: all logic handled in timer to reduce network load.
  }

bool EnsureConnection()
  {
   if(InpUseHttp)
      return true;
   socketClient.Close();
   if(!socketClient.SocketCreate())
     {
      Print("Natron EA: SocketCreate failed. Error ", GetLastError());
      return false;
     }
   if(!socketClient.Connect(InpHost, InpTcpPort))
     {
      Print("Natron EA: Connect failed. Error ", GetLastError());
      socketClient.Close();
      return false;
     }
   Print("Natron EA: Connected to ", InpHost, ":", InpTcpPort);
   return true;
  }

bool SendRequest(string payload, string &response)
  {
   if(InpUseHttp)
      return SendHttpRequest(payload, response);
   return SendTcpRequest(payload + "\n", response);
  }

bool SendHttpRequest(string payload, string &response)
  {
   string url = "http://" + InpHost + ":" + IntegerToString(InpHttpPort) + "/predict";
   char headers[];
   string headerString = "Content-Type: application/json\r\n";
   StringToCharArray(headerString, headers);

   char data[];
   StringToCharArray(payload, data, 0, WHOLE_ARRAY, CP_UTF8);
   char result[];
   string cookie = "";
   string referer = "";

   int status = WebRequest("POST", url, headers, 5000, data, result, cookie, referer);
   if(status != 200)
     {
      PrintFormat("Natron EA: WebRequest failed. Status=%d", status);
      return false;
     }
   response = CharArrayToString(result);
   return true;
  }

bool SendTcpRequest(string payload, string &response)
  {
   if(!socketClient.IsConnected())
     {
      Print("Natron EA: socket not connected, retrying...");
      if(!EnsureConnection())
         return false;
     }
   uchar data[];
   StringToCharArray(payload, data, 0, WHOLE_ARRAY, CP_UTF8);
   int sent = socketClient.Send(data, ArraySize(data));
   if(sent <= 0)
     {
      Print("Natron EA: send failed, reconnecting");
      socketClient.Close();
      return false;
     }

   uchar buffer[8192];
   ArrayInitialize(buffer, 0);
   int received = socketClient.Receive(buffer, ArraySize(buffer), 0, 2000);
   if(received <= 0)
     {
      Print("Natron EA: receive timeout");
      socketClient.Close();
      return false;
     }
   response = CharArrayToString(buffer, 0, received);
   return true;
  }

string BuildPayload()
  {
   MqlRates rates[];
   if(CopyRates(_Symbol, _Period, 0, InpSequenceLength, rates) <= 0)
      return "";
   ArraySetAsSeries(rates, true);

   string json = "{\"candles\":[";
   for(int i = InpSequenceLength - 1; i >= 0; --i)
     {
      string entry = StringFormat(
         "{\"time\":\"%s\",\"open\":%.6f,\"high\":%.6f,\"low\":%.6f,\"close\":%.6f,\"volume\":%.2f}",
         TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
         rates[i].open,
         rates[i].high,
         rates[i].low,
         rates[i].close,
         rates[i].tick_volume
      );
      json += entry;
      if(i != 0)
         json += ",";
     }
   json += "]}";
   return json;
  }

double ExtractJsonDouble(const string json, const string key)
  {
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0)
      return 0.0;
   int start = pos + StringLen(pattern);
   int end = start;
   int total = StringLen(json);
   while(end < total)
     {
      int ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']')
         break;
      end++;
     }
   string value = StringTrim(StringSubstr(json, start, end - start));
   return StringToDouble(value);
  }

string ExtractJsonString(const string json, const string key)
  {
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0)
      return "";
   int start = pos + StringLen(pattern);
   int firstQuote = StringFind(json, "\"", start);
   if(firstQuote < 0)
      return "";
   int secondQuote = StringFind(json, "\"", firstQuote + 1);
   if(secondQuote < 0)
      return "";
   return StringSubstr(json, firstQuote + 1, secondQuote - firstQuote - 1);
  }

void UpdateChart()
  {
   string text = StringFormat(
      "Natron Transformer\nBuy Prob: %.2f\nSell Prob: %.2f\nDirection Up: %.2f\nRegime: %s",
      lastBuyProb,
      lastSellProb,
      lastDirectionUp,
      lastRegime
   );
   Comment(text);
  }

void HandleSignals()
  {
   if(lastBuyProb >= InpBuyThreshold && !PositionsExist(POSITION_TYPE_BUY))
     {
      trade.Buy(InpLots, _Symbol, 0, 0, 0, "Natron Buy");
     }
   if(lastSellProb >= InpSellThreshold && !PositionsExist(POSITION_TYPE_SELL))
     {
      trade.Sell(InpLots, _Symbol, 0, 0, 0, "Natron Sell");
     }
  }

bool PositionsExist(int type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; --i)
     {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
        {
         if(PositionGetInteger(POSITION_MAGIC) == InpMagic &&
            PositionGetInteger(POSITION_TYPE) == type)
            return true;
        }
     }
   return false;
  }
