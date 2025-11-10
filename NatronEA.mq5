//+------------------------------------------------------------------+
//|                                                    NatronEA.mq5 |
//|                                  Copyright 2024, Natron AI Team |
//|                                             https://natron.ai  |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property link      "https://github.com/natron-ai"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerIP = "127.0.0.1";        // Python Server IP
input int      ServerPort = 8888;            // Python Server Port (Socket) or 5000 (HTTP)
input bool     UseSocket = true;              // Use Socket (true) or HTTP (false)
input int      MagicNumber = 123456;          // Magic Number
input double   LotSize = 0.01;                // Lot Size
input int      StopLoss = 100;                // Stop Loss (points)
input int      TakeProfit = 200;              // Take Profit (points)
input int      SequenceLength = 96;           // Sequence Length (candles)
input ENUM_TIMEFRAMES Timeframe = PERIOD_M15; // Timeframe
input double   BuyThreshold = 0.6;           // Buy Signal Threshold
input double   SellThreshold = 0.6;          // Sell Signal Threshold
input bool     UseRegimeFilter = true;        // Use Regime Filter
input string   AllowedRegimes = "BULL_STRONG,BULL_WEAK"; // Allowed Regimes

//--- Global variables
CTrade trade;
int socket_handle = INVALID_HANDLE;
datetime last_bar_time = 0;
string allowed_regimes_array[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   if(UseRegimeFilter)
   {
      int count = StringSplit(AllowedRegimes, ',', allowed_regimes_array);
      Print("Allowed regimes: ", count);
   }
   
   if(UseSocket)
   {
      socket_handle = SocketCreate();
      if(socket_handle == INVALID_HANDLE)
      {
         Print("Failed to create socket");
         return(INIT_FAILED);
      }
      
      if(!SocketConnect(socket_handle, ServerIP, ServerPort, 1000))
      {
         Print("Failed to connect to socket server ", ServerIP, ":", ServerPort);
         SocketClose(socket_handle);
         return(INIT_FAILED);
      }
      Print("Connected to Natron socket server");
   }
   
   Print("Server: ", ServerIP, ":", ServerPort);
   Print("Timeframe: ", EnumToString(Timeframe));
   Print("Sequence Length: ", SequenceLength);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(socket_handle != INVALID_HANDLE)
   {
      SocketClose(socket_handle);
   }
   Print("Natron EA disconnected");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   datetime current_bar_time = iTime(_Symbol, Timeframe, 0);
   if(current_bar_time == last_bar_time) return;
   last_bar_time = current_bar_time;
   
   if(Bars(_Symbol, Timeframe) < SequenceLength + 10)
   {
      Print("Not enough bars");
      return;
   }
   
   string json_data = CollectCandlesJSON();
   if(json_data == "") return;
   
   string response = "";
   if(UseSocket)
   {
      response = SendSocketRequest(json_data);
   }
   else
   {
      response = SendHTTPRequest(json_data);
   }
   
   if(response == "") return;
   
   ParseAndExecute(response);
}

//+------------------------------------------------------------------+
//| Collect candles and format as JSON                               |
//+------------------------------------------------------------------+
string CollectCandlesJSON()
{
   string json = "{\"candles\":[";
   
   for(int i = SequenceLength - 1; i >= 0; i--)
   {
      datetime time = iTime(_Symbol, Timeframe, i);
      double open = iOpen(_Symbol, Timeframe, i);
      double high = iHigh(_Symbol, Timeframe, i);
      double low = iLow(_Symbol, Timeframe, i);
      double close = iClose(_Symbol, Timeframe, i);
      long volume = iVolume(_Symbol, Timeframe, i);
      
      MqlDateTime dt;
      TimeToStruct(time, dt);
      string time_str = StringFormat("%04d-%02d-%02d %02d:%02d:00", 
                                     dt.year, dt.mon, dt.day, dt.hour, dt.min);
      
      if(i < SequenceLength - 1) json += ",";
      
      json += StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%lld}",
         time_str, open, high, low, close, volume
      );
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Send socket request                                              |
//+------------------------------------------------------------------+
string SendSocketRequest(string json_data)
{
   char send_buffer[];
   StringToCharArray(json_data + "\n", send_buffer);
   
   if(SocketSend(socket_handle, send_buffer) <= 0)
   {
      Print("Failed to send socket request");
      return "";
   }
   
   char recv_buffer[];
   string response = "";
   int timeout = 5000;
   int start_time = GetTickCount();
   
   while(GetTickCount() - start_time < timeout)
   {
      int received = SocketRead(socket_handle, recv_buffer, 0, 1000);
      if(received > 0)
      {
         response += CharArrayToString(recv_buffer, 0, received);
         if(StringFind(response, "\n") >= 0)
         {
            int newline_pos = StringFind(response, "\n");
            return StringSubstr(response, 0, newline_pos);
         }
      }
      Sleep(10);
   }
   
   return response;
}

//+------------------------------------------------------------------+
//| Send HTTP request                                                |
//+------------------------------------------------------------------+
string SendHTTPRequest(string json_data)
{
   string request = "POST /predict HTTP/1.1\r\n";
   request += "Host: " + ServerIP + ":" + IntegerToString(ServerPort) + "\r\n";
   request += "Content-Type: application/json\r\n";
   request += "Content-Length: " + IntegerToString(StringLen(json_data)) + "\r\n";
   request += "Connection: close\r\n\r\n";
   request += json_data;
   
   char send_buffer[];
   StringToCharArray(request, send_buffer);
   
   int http_socket = SocketCreate();
   if(http_socket == INVALID_HANDLE) return "";
   
   if(!SocketConnect(http_socket, ServerIP, ServerPort, 1000))
   {
      SocketClose(http_socket);
      return "";
   }
   
   SocketSend(http_socket, send_buffer);
   
   char recv_buffer[];
   string response = "";
   int timeout = 5000;
   int start_time = GetTickCount();
   
   while(GetTickCount() - start_time < timeout)
   {
      int received = SocketRead(http_socket, recv_buffer, 0, 1000);
      if(received > 0)
      {
         response += CharArrayToString(recv_buffer, 0, received);
         if(StringFind(response, "\r\n\r\n") >= 0)
         {
            int body_start = StringFind(response, "\r\n\r\n") + 4;
            SocketClose(http_socket);
            return StringSubstr(response, body_start);
         }
      }
      Sleep(10);
   }
   
   SocketClose(http_socket);
   return response;
}

//+------------------------------------------------------------------+
//| Parse JSON response and execute trades                           |
//+------------------------------------------------------------------+
void ParseAndExecute(string json_response)
{
   double buy_prob = ExtractDouble(json_response, "buy_prob");
   double sell_prob = ExtractDouble(json_response, "sell_prob");
   string regime = ExtractString(json_response, "regime");
   
   if(buy_prob == -1 || sell_prob == -1)
   {
      Print("Failed to parse response");
      return;
   }
   
   Print("Signal - Buy: ", buy_prob, ", Sell: ", sell_prob, ", Regime: ", regime);
   
   if(UseRegimeFilter)
   {
      bool regime_allowed = false;
      for(int i = 0; i < ArraySize(allowed_regimes_array); i++)
      {
         if(regime == allowed_regimes_array[i])
         {
            regime_allowed = true;
            break;
         }
      }
      if(!regime_allowed)
      {
         Print("Regime ", regime, " not allowed");
         return;
      }
   }
   
   bool has_buy = PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY;
   bool has_sell = PositionSelect(_Symbol) && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL;
   
   if(buy_prob >= BuyThreshold && !has_buy)
   {
      if(has_sell) trade.PositionClose(_Symbol);
      
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - StopLoss * _Point;
      double tp = price + TakeProfit * _Point;
      
      if(trade.Buy(LotSize, _Symbol, price, sl, tp, "Natron Buy"))
      {
         Print("Buy opened: ", price);
      }
   }
   else if(sell_prob >= SellThreshold && !has_sell)
   {
      if(has_buy) trade.PositionClose(_Symbol);
      
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + StopLoss * _Point;
      double tp = price - TakeProfit * _Point;
      
      if(trade.Sell(LotSize, _Symbol, price, sl, tp, "Natron Sell"))
      {
         Print("Sell opened: ", price);
      }
   }
}

//+------------------------------------------------------------------+
//| Extract double from JSON                                         |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return -1;
   
   int start = pos + StringLen(search);
   int end = StringFind(json, ",", start);
   if(end < 0) end = StringFind(json, "}", start);
   if(end < 0) return -1;
   
   string value = StringSubstr(json, start, end - start);
   StringReplace(value, " ", "");
   return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| Extract string from JSON                                         |
//+------------------------------------------------------------------+
string ExtractString(string json, string key)
{
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   
   int start = pos + StringLen(search);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   
   return StringSubstr(json, start, end - start);
}
