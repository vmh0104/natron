//+------------------------------------------------------------------+
//|                                              natron_ea.mq5      |
//|                        Natron AI Trading System - MQL5 EA       |
//+------------------------------------------------------------------+
#property copyright "Natron V1.0"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input int      SocketPort = 8888;        // Python server port
input string   SymbolName = "EURUSD";    // Trading symbol
input double   LotSize = 0.01;           // Lot size
input int      MagicNumber = 123456;     // Magic number

//--- Global variables
CTrade trade;
int socket_handle = INVALID_HANDLE;
string socket_buffer = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set trade parameters
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Connect to Python server socket
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return(INIT_FAILED);
   }
   
   if(!SocketConnect(socket_handle, "localhost", SocketPort, 1000))
   {
      Print("Failed to connect to Python server");
      SocketClose(socket_handle);
      return(INIT_FAILED);
   }
   
   Print("Natron EA initialized successfully");
   Print("Connected to Python server on port ", SocketPort);
   
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
   Print("Natron EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for incoming signals from Python server
   CheckForSignals();
   
   // Manage existing positions (trailing stop, etc.)
   ManagePositions();
}

//+------------------------------------------------------------------+
//| Check for trading signals from Python server                     |
//+------------------------------------------------------------------+
void CheckForSignals()
{
   if(socket_handle == INVALID_HANDLE)
      return;
   
   // Read data from socket
   char data[];
   int received = SocketRead(socket_handle, data, 1024, 100);
   
   if(received > 0)
   {
      string message = CharArrayToString(data);
      ProcessSignal(message);
   }
}

//+------------------------------------------------------------------+
//| Process trading signal from Python                               |
//+------------------------------------------------------------------+
void ProcessSignal(string message)
{
   // Format: "ACTION|SYMBOL|PRICE|STOP_LOSS|TAKE_PROFIT|COMMENT"
   string parts[];
   int count = StringSplit(message, '|', parts);
   
   if(count < 3)
   {
      Print("Invalid signal format: ", message);
      return;
   }
   
   string action = parts[0];
   string symbol = parts[1];
   double price = StringToDouble(parts[2]);
   double stop_loss = (count > 3) ? StringToDouble(parts[3]) : 0;
   double take_profit = (count > 4) ? StringToDouble(parts[4]) : 0;
   string comment = (count > 5) ? parts[5] : "Natron";
   
   // Normalize price
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   
   if(action == "BUY")
   {
      // Close opposite positions
      ClosePositions(symbol, POSITION_TYPE_SELL);
      
      // Open buy position
      if(!PositionSelect(symbol))
      {
         trade.Buy(LotSize, symbol, ask, stop_loss, take_profit, comment);
         Print("Opened BUY position: ", symbol, " at ", ask);
      }
   }
   else if(action == "SELL")
   {
      // Close opposite positions
      ClosePositions(symbol, POSITION_TYPE_BUY);
      
      // Open sell position
      if(!PositionSelect(symbol))
      {
         trade.Sell(LotSize, symbol, bid, stop_loss, take_profit, comment);
         Print("Opened SELL position: ", symbol, " at ", bid);
      }
   }
   else if(action == "CLOSE")
   {
      // Close all positions
      CloseAllPositions(symbol);
      Print("Closed all positions for ", symbol);
   }
}

//+------------------------------------------------------------------+
//| Close positions of specified type                                |
//+------------------------------------------------------------------+
void ClosePositions(string symbol, ENUM_POSITION_TYPE type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
            PositionGetInteger(POSITION_TYPE) == type)
         {
            trade.PositionClose(ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close all positions for symbol                                   |
//+------------------------------------------------------------------+
void CloseAllPositions(string symbol)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            trade.PositionClose(ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Manage existing positions (trailing stop, etc.)                  |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == SymbolName &&
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            // Implement trailing stop logic here
            // Example: Update stop loss based on price movement
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Send candle data to Python server                                |
//+------------------------------------------------------------------+
void SendCandleData()
{
   if(socket_handle == INVALID_HANDLE)
      return;
   
   // Get current candle data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(SymbolName, PERIOD_CURRENT, 0, 1, rates);
   
   if(copied > 0)
   {
      // Format: JSON with OHLCV
      string json = StringFormat(
         "{\"time\":%d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%d}",
         rates[0].time,
         rates[0].open,
         rates[0].high,
         rates[0].low,
         rates[0].close,
         rates[0].tick_volume
      );
      
      // Send to Python server
      uchar data[];
      StringToCharArray(json, data);
      SocketSend(socket_handle, data);
   }
}
