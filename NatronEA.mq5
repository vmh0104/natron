//+------------------------------------------------------------------+
//|                                              NatronEA.mq5        |
//|                        Natron Transformer Trading EA             |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Natron AI"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerIP = "127.0.0.1";        // Python Server IP
input int      ServerPort = 8888;             // Python Server Port
input int      SequenceLength = 96;           // Sequence length (must match model)
input int      MagicNumber = 123456;          // Magic number
input double   LotSize = 0.01;                // Lot size
input int      StopLoss = 100;                // Stop Loss (points)
input int      TakeProfit = 200;              // Take Profit (points)
input double   BuyThreshold = 0.6;            // Buy signal threshold
input double   SellThreshold = 0.6;           // Sell signal threshold
input int      RequestTimeout = 5000;          // Request timeout (ms)
input bool     EnableTrading = true;          // Enable trading
input ENUM_TIMEFRAMES Timeframe = PERIOD_M15; // Timeframe

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
   // Set trade parameters
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);
   
   // Connect to Python server
   if(!ConnectToServer())
   {
      Print("Failed to connect to Python server. EA will retry on next tick.");
      return(INIT_SUCCEEDED);
   }
   
   Print("Natron EA initialized successfully");
   Print("Server: ", ServerIP, ":", ServerPort);
   Print("Timeframe: ", EnumToString(Timeframe));
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   DisconnectFromServer();
   Print("Natron EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, Timeframe, 0);
   if(currentBarTime == lastBarTime)
      return;
   
   lastBarTime = currentBarTime;
   
   // Ensure we have enough bars
   if(Bars(_Symbol, Timeframe) < SequenceLength)
   {
      Print("Not enough bars. Need ", SequenceLength, " bars.");
      return;
   }
   
   // Collect OHLCV data
   MqlRates rates[];
   if(CopyRates(_Symbol, Timeframe, 0, SequenceLength, rates) < SequenceLength)
   {
      Print("Failed to copy rates");
      return;
   }
   
   // Prepare JSON request
   string jsonRequest = PrepareRequest(rates);
   
   // Send request to Python server
   string jsonResponse = SendRequest(jsonRequest);
   
   if(jsonResponse == "")
   {
      Print("Failed to get response from server");
      // Try to reconnect
      if(socketHandle == INVALID_HANDLE)
         ConnectToServer();
      return;
   }
   
   // Parse response
   if(!ParseResponse(jsonResponse))
   {
      Print("Failed to parse response: ", jsonResponse);
      return;
   }
   
   // Execute trading logic
   if(EnableTrading)
      ExecuteTrading();
}

//+------------------------------------------------------------------+
//| Connect to Python server                                         |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return false;
   }
   
   if(!SocketConnect(socketHandle, ServerIP, ServerPort, RequestTimeout))
   {
      Print("Failed to connect to ", ServerIP, ":", ServerPort);
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return false;
   }
   
   Print("Connected to Python server");
   return true;
}

//+------------------------------------------------------------------+
//| Disconnect from server                                           |
//+------------------------------------------------------------------+
void DisconnectFromServer()
{
   if(socketHandle != INVALID_HANDLE)
   {
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
//| Prepare JSON request                                             |
//+------------------------------------------------------------------+
string PrepareRequest(MqlRates &rates[])
{
   string json = "{\"action\":\"predict\",\"candles\":[";
   
   for(int i = ArraySize(rates) - 1; i >= 0; i--)
   {
      if(i < ArraySize(rates) - 1)
         json += ",";
      
      json += "{";
      json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, 5) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, 5) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, 5) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, 5) + ",";
      json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Send request to server                                           |
//+------------------------------------------------------------------+
string SendRequest(string request)
{
   if(socketHandle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return "";
   }
   
   // Send request
   uchar sendBuffer[];
   StringToCharArray(request, sendBuffer, 0, StringLen(request));
   
   if(SocketSend(socketHandle, sendBuffer) <= 0)
   {
      Print("Failed to send request");
      DisconnectFromServer();
      return "";
   }
   
   // Receive response
   uchar receiveBuffer[];
   string response = "";
   uint timeout = GetTickCount() + RequestTimeout;
   
   while(GetTickCount() < timeout)
   {
      int received = SocketRead(socketHandle, receiveBuffer, 0, 1000);
      if(received > 0)
      {
         response += CharArrayToString(receiveBuffer, 0, received);
         // Check if we have complete JSON
         if(StringFind(response, "}") >= 0)
            break;
      }
      Sleep(10);
   }
   
   return response;
}

//+------------------------------------------------------------------+
//| Parse JSON response                                              |
//+------------------------------------------------------------------+
bool ParseResponse(string jsonResponse)
{
   // Simple JSON parsing (for production, use proper JSON library)
   int statusPos = StringFind(jsonResponse, "\"status\"");
   if(statusPos < 0)
      return false;
   
   int statusStart = StringFind(jsonResponse, ":", statusPos) + 1;
   int statusEnd = StringFind(jsonResponse, ",", statusStart);
   if(statusEnd < 0)
      statusEnd = StringFind(jsonResponse, "}", statusStart);
   
   string status = StringSubstr(jsonResponse, statusStart, statusEnd - statusStart);
   status = StringTrimLeft(StringTrimRight(status));
   status = StringSubstr(status, 1, StringLen(status) - 2); // Remove quotes
   
   if(status != "success")
   {
      Print("Server returned error status: ", status);
      return false;
   }
   
   // Extract buy_prob
   int buyPos = StringFind(jsonResponse, "\"buy_prob\"");
   if(buyPos >= 0)
   {
      int buyStart = StringFind(jsonResponse, ":", buyPos) + 1;
      int buyEnd = StringFind(jsonResponse, ",", buyStart);
      if(buyEnd < 0)
         buyEnd = StringFind(jsonResponse, "}", buyStart);
      string buyStr = StringSubstr(jsonResponse, buyStart, buyEnd - buyStart);
      lastBuyProb = StringToDouble(StringTrimLeft(StringTrimRight(buyStr)));
   }
   
   // Extract sell_prob
   int sellPos = StringFind(jsonResponse, "\"sell_prob\"");
   if(sellPos >= 0)
   {
      int sellStart = StringFind(jsonResponse, ":", sellPos) + 1;
      int sellEnd = StringFind(jsonResponse, ",", sellStart);
      if(sellEnd < 0)
         sellEnd = StringFind(jsonResponse, "}", sellStart);
      string sellStr = StringSubstr(jsonResponse, sellStart, sellEnd - sellStart);
      lastSellProb = StringToDouble(StringTrimLeft(StringTrimRight(sellStr)));
   }
   
   // Extract regime
   int regimePos = StringFind(jsonResponse, "\"regime\"");
   if(regimePos >= 0)
   {
      int regimeStart = StringFind(jsonResponse, ":", regimePos) + 1;
      int regimeEnd = StringFind(jsonResponse, ",", regimeStart);
      if(regimeEnd < 0)
         regimeEnd = StringFind(jsonResponse, "}", regimeStart);
      string regimeStr = StringSubstr(jsonResponse, regimeStart, regimeEnd - regimeStart);
      regimeStr = StringTrimLeft(StringTrimRight(regimeStr));
      lastRegime = StringSubstr(regimeStr, 1, StringLen(regimeStr) - 2); // Remove quotes
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic                                            |
//+------------------------------------------------------------------+
void ExecuteTrading()
{
   // Check existing positions
   bool hasBuy = PositionSelect(_Symbol);
   if(hasBuy)
   {
      long posType = PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY)
         hasBuy = true;
      else
         hasBuy = false;
   }
   else
      hasBuy = false;
   
   bool hasSell = PositionSelect(_Symbol);
   if(hasSell)
   {
      long posType = PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_SELL)
         hasSell = true;
      else
         hasSell = false;
   }
   else
      hasSell = false;
   
   // Buy signal
   if(lastBuyProb >= BuyThreshold && !hasBuy)
   {
      // Close sell position if exists
      if(hasSell)
         trade.PositionClose(_Symbol);
      
      // Open buy position
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - StopLoss * _Point;
      double tp = price + TakeProfit * _Point;
      
      if(trade.Buy(LotSize, _Symbol, price, sl, tp, "Natron Buy Signal"))
      {
         Print("BUY order opened. Buy prob: ", lastBuyProb, ", Regime: ", lastRegime);
      }
   }
   
   // Sell signal
   if(lastSellProb >= SellThreshold && !hasSell)
   {
      // Close buy position if exists
      if(hasBuy)
         trade.PositionClose(_Symbol);
      
      // Open sell position
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + StopLoss * _Point;
      double tp = price - TakeProfit * _Point;
      
      if(trade.Sell(LotSize, _Symbol, price, sl, tp, "Natron Sell Signal"))
      {
         Print("SELL order opened. Sell prob: ", lastSellProb, ", Regime: ", lastRegime);
      }
   }
}

//+------------------------------------------------------------------+
