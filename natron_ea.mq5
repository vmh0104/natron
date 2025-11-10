//+------------------------------------------------------------------+
//|                                                   natron_ea.mq5 |
//|   Natron V2 Expert Advisor - connects MetaTrader 5 to Natron AI |
//+------------------------------------------------------------------+
#property copyright ""
#property version   "1.00"
#property strict

#include <stderror.mqh>
#include <stdlib.mqh>

input string  InpServerHost      = "127.0.0.1";
input int     InpServerPort      = 8765;
input int     InpTimeoutMs       = 1000;
input double  InpBuyThreshold    = 0.65;
input double  InpSellThreshold   = 0.65;
input int     InpSequenceLength  = 120;

int            g_socket = INVALID_HANDLE;
CJAVal         g_response;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!SocketConnect())
   {
      Print("Natron EA: Unable to connect to server.");
      return(INIT_FAILED);
   }
   Print("Natron EA: Connected to Natron server ", InpServerHost, ":", InpServerPort);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_socket != INVALID_HANDLE)
   {
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
   }
   Comment("");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!EnsureConnection())
      return;

   string payload = BuildPayload();
   if(payload == "")
      return;

   if(!SocketSendPayload(payload))
      return;

   string response = SocketReceiveResponse();
   if(response == "")
      return;

   if(!g_response.Deserialize(response))
   {
      Print("Natron EA: Failed to parse JSON response: ", response);
      return;
   }

   double buyProb = g_response["buy_prob"];
   double sellProb = g_response["sell_prob"];
   double upProb = g_response["direction_up"];
   string regime = (string)g_response["regime"];
   double confidence = g_response["confidence"];

   Comment(
      "Natron AI\n",
      "Buy Prob: ", DoubleToString(buyProb, 3), "\n",
      "Sell Prob: ", DoubleToString(sellProb, 3), "\n",
      "Up Prob: ", DoubleToString(upProb, 3), "\n",
      "Regime: ", regime, "\n",
      "Confidence: ", DoubleToString(confidence, 3)
   );

   ManagePositions(buyProb, sellProb);
}

//+------------------------------------------------------------------+
//| Socket helpers                                                    |
//+------------------------------------------------------------------+
bool SocketConnect()
{
   if(g_socket != INVALID_HANDLE)
      SocketClose(g_socket);

   g_socket = SocketCreate();
   if(g_socket == INVALID_HANDLE)
   {
      Print("Natron EA: SocketCreate failed. Error ", GetLastError());
      return(false);
   }

   if(!SocketConnect(g_socket, InpServerHost, (ushort)InpServerPort, InpTimeoutMs))
   {
      Print("Natron EA: SocketConnect failed. Error ", GetLastError());
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
      return(false);
   }
   return(true);
}

bool EnsureConnection()
{
   if(g_socket == INVALID_HANDLE)
      return(SocketConnect());
   return(true);
}

bool SocketSendPayload(const string payload)
{
   uchar buffer[];
   string message = payload + "\n";
   int size = StringToCharArray(message, buffer, 0, WHOLE_ARRAY, CP_UTF8);
   if(size <= 0)
      return(false);
   int sent = SocketSend(g_socket, buffer, size - 1);
   if(sent != size - 1)
   {
      Print("Natron EA: SocketSend failed. Error ", GetLastError());
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
      return(false);
   }
   return(true);
}

string SocketReceiveResponse()
{
   uchar buffer[4096];
   int received = SocketRead(g_socket, buffer, ArraySize(buffer), InpTimeoutMs);
   if(received <= 0)
   {
      Print("Natron EA: SocketRead failed. Error ", GetLastError());
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
      return("");
   }
   return CharArrayToString(buffer, 0, received, CP_UTF8);
}

//+------------------------------------------------------------------+
//| Payload creation                                                 |
//+------------------------------------------------------------------+
string BuildPayload()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, InpSequenceLength, rates);
   if(copied <= 0)
   {
      Print("Natron EA: Failed to copy rates. Error ", GetLastError());
      return("");
   }

   string json = "{\"ohlcv\":[";
   for(int i = copied - 1; i >= 0; i--)
   {
      datetime t = rates[i].time;
      string timestamp = TimeToString(t, TIME_DATE|TIME_MINUTES|TIME_SECONDS);
      string entry;
      StringFormat(entry,
         "{\"time\":\"%s\",\"open\":%G,\"high\":%G,\"low\":%G,\"close\":%G,\"volume\":%G}",
         timestamp,
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

//+------------------------------------------------------------------+
//| Position management                                              |
//+------------------------------------------------------------------+
void ManagePositions(double buyProb, double sellProb)
{
   int totalPositions = PositionsTotal();
   bool hasBuy = false;
   bool hasSell = false;
   for(int i = 0; i < totalPositions; i++)
   {
      if(!PositionSelectByIndex(i))
         continue;
      if(PositionGetSymbol(i) != _Symbol)
         continue;
      long type = PositionGetInteger(POSITION_TYPE);
      if(type == POSITION_TYPE_BUY)
         hasBuy = true;
      if(type == POSITION_TYPE_SELL)
         hasSell = true;
   }

   if(buyProb >= InpBuyThreshold && !hasBuy)
   {
      CloseSellPositions();
      OpenPosition(ORDER_TYPE_BUY);
   }
   else if(sellProb >= InpSellThreshold && !hasSell)
   {
      CloseBuyPositions();
      OpenPosition(ORDER_TYPE_SELL);
   }
}

void OpenPosition(const ENUM_ORDER_TYPE orderType)
{
   double lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double price = (orderType == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   MqlTradeRequest request;
   ZeroMemory(request);
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = orderType;
   request.price = price;
   request.deviation = 10;
   request.magic = 987654;
   request.comment = "NatronAI";

   MqlTradeResult result;
   if(!OrderSend(request, result))
      Print("Natron EA: OrderSend failed. Error ", GetLastError());
}

void CloseBuyPositions()
{
   ClosePositionsOfType(POSITION_TYPE_BUY);
}

void CloseSellPositions()
{
   ClosePositionsOfType(POSITION_TYPE_SELL);
}

void ClosePositionsOfType(const long type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!PositionSelectByIndex(i))
         continue;
      if(PositionGetSymbol(i) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_TYPE) != type)
         continue;

      ulong ticket = PositionGetInteger(POSITION_TICKET);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      MqlTradeRequest request;
      ZeroMemory(request);
      request.action = TRADE_ACTION_DEAL;
      request.position = ticket;
      request.symbol = _Symbol;
      request.volume = volume;
      request.type = (type == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      request.price = price;
      request.deviation = 10;
      request.magic = 987654;

      MqlTradeResult result;
      if(!OrderSend(request, result))
         Print("Natron EA: Failed to close position ", ticket, " Error ", GetLastError());
   }
}
