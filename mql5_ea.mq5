//+------------------------------------------------------------------+
//|                                          Natron_EA.mq5           |
//|                        Natron Transformer Trading EA             |
//+------------------------------------------------------------------+
#property copyright "Natron AI"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "localhost";      // Python Server Host
input int      ServerPort = 8888;            // Python Server Port
input double   LotSize = 0.01;                // Lot Size
input int      MagicNumber = 12345;            // Magic Number
input int      StopLoss = 50;                  // Stop Loss (points)
input int      TakeProfit = 100;               // Take Profit (points)
input bool     UseStopLoss = true;             // Use Stop Loss
input bool     UseTakeProfit = true;           // Use Take Profit
input double   BuyThreshold = 0.6;            // Buy Signal Threshold
input double   SellThreshold = 0.6;            // Sell Signal Threshold
input int      CandleCount = 96;               // Number of candles for prediction

//--- Global variables
int socketHandle = INVALID_HANDLE;
CTrade trade;
datetime lastBarTime = 0;
double lastBuyProb = 0.0;
double lastSellProb = 0.0;
string lastRegime = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Connect to Python server
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return(INIT_FAILED);
   }
   
   if(!SocketConnect(socketHandle, ServerHost, ServerPort, 5000))
   {
      Print("Failed to connect to server: ", ServerHost, ":", ServerPort);
      SocketClose(socketHandle);
      return(INIT_FAILED);
   }
   
   Print("Connected to Natron AI server");
   
   // Send initial ping
   SendPing();
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(socketHandle != INVALID_HANDLE)
   {
      SocketClose(socketHandle);
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBarTime != lastBarTime)
   {
      lastBarTime = currentBarTime;
      OnNewBar();
   }
   
   // Manage existing positions
   ManagePositions();
}

//+------------------------------------------------------------------+
//| New bar event                                                     |
//+------------------------------------------------------------------+
void OnNewBar()
{
   // Collect candle data
   MqlRates rates[];
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, CandleCount, rates) < CandleCount)
   {
      Print("Failed to copy rates");
      return;
   }
   
   // Prepare JSON message
   string json = PrepareCandleJSON(rates);
   
   // Send prediction request
   string response = SendPredictionRequest(json);
   
   if(response != "")
   {
      ProcessPrediction(response);
   }
}

//+------------------------------------------------------------------+
//| Prepare candle data as JSON                                      |
//+------------------------------------------------------------------+
string PrepareCandleJSON(MqlRates &rates[])
{
   string json = "{\"type\":\"PREDICT\",\"candles\":[";
   
   for(int i = ArraySize(rates) - 1; i >= 0; i--)
   {
      if(i < ArraySize(rates) - 1) json += ",";
      
      json += "{";
      json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Send prediction request                                          |
//+------------------------------------------------------------------+
string SendPredictionRequest(string json)
{
   if(socketHandle == INVALID_HANDLE)
   {
      // Try to reconnect
      socketHandle = SocketCreate();
      if(socketHandle == INVALID_HANDLE || !SocketConnect(socketHandle, ServerHost, ServerPort, 5000))
      {
         Print("Failed to reconnect to server");
         return "";
      }
   }
   
   // Send request
   char request[];
   StringToCharArray(json, request, 0, StringLen(json));
   
   if(SocketSend(socketHandle, request) <= 0)
   {
      Print("Failed to send request");
      return "";
   }
   
   // Receive response
   char response[];
   int timeout = 5000; // 5 seconds
   int received = SocketReceive(socketHandle, response, timeout);
   
   if(received <= 0)
   {
      Print("Failed to receive response or timeout");
      return "";
   }
   
   return CharArrayToString(response, 0, received);
}

//+------------------------------------------------------------------+
//| Process prediction response                                      |
//+------------------------------------------------------------------+
void ProcessPrediction(string response)
{
   // Parse JSON (simplified - in production use proper JSON parser)
   int buyPos = StringFind(response, "\"buy_prob\":");
   int sellPos = StringFind(response, "\"sell_prob\":");
   int regimePos = StringFind(response, "\"regime\":\"");
   
   if(buyPos >= 0)
   {
      string buyStr = StringSubstr(response, buyPos + 11, 10);
      lastBuyProb = StringToDouble(buyStr);
   }
   
   if(sellPos >= 0)
   {
      string sellStr = StringSubstr(response, sellPos + 12, 10);
      lastSellProb = StringToDouble(sellStr);
   }
   
   if(regimePos >= 0)
   {
      int regimeEnd = StringFind(response, "\"", regimePos + 10);
      if(regimeEnd > regimePos)
      {
         lastRegime = StringSubstr(response, regimePos + 10, regimeEnd - regimePos - 10);
      }
   }
   
   // Execute trades based on signals
   if(lastBuyProb >= BuyThreshold && !HasPosition(POSITION_TYPE_BUY))
   {
      // Close sell positions first
      if(HasPosition(POSITION_TYPE_SELL))
      {
         CloseAllPositions();
      }
      
      // Open buy position
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = UseStopLoss ? ask - StopLoss * _Point : 0;
      double tp = UseTakeProfit ? ask + TakeProfit * _Point : 0;
      
      if(trade.Buy(LotSize, _Symbol, ask, sl, tp, "Natron Buy Signal"))
      {
         Print("Buy order opened. Prob: ", lastBuyProb, " Regime: ", lastRegime);
      }
   }
   else if(lastSellProb >= SellThreshold && !HasPosition(POSITION_TYPE_SELL))
   {
      // Close buy positions first
      if(HasPosition(POSITION_TYPE_BUY))
      {
         CloseAllPositions();
      }
      
      // Open sell position
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = UseStopLoss ? bid + StopLoss * _Point : 0;
      double tp = UseTakeProfit ? bid - TakeProfit * _Point : 0;
      
      if(trade.Sell(LotSize, _Symbol, bid, sl, tp, "Natron Sell Signal"))
      {
         Print("Sell order opened. Prob: ", lastSellProb, " Regime: ", lastRegime);
      }
   }
}

//+------------------------------------------------------------------+
//| Check if position exists                                         |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == type)
         {
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            trade.PositionClose(ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage existing positions                                        |
//+------------------------------------------------------------------+
void ManagePositions()
{
   // Add trailing stop, break-even, or other management logic here
   // This is a placeholder for position management
}

//+------------------------------------------------------------------+
//| Send ping to server                                              |
//+------------------------------------------------------------------+
void SendPing()
{
   string ping = "{\"type\":\"PING\"}";
   char request[];
   StringToCharArray(ping, request, 0, StringLen(ping));
   SocketSend(socketHandle, request);
}

//+------------------------------------------------------------------+
