//+------------------------------------------------------------------+
//|                                            natron_ea.mq5         |
//|                        Natron AI Trading System - MQL5 EA        |
//+------------------------------------------------------------------+
#property copyright "Natron Trading System"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string ServerHost = "localhost";      // Python server host
input int    ServerPort = 8888;             // Python server port
input double LotSize = 0.01;                // Lot size
input int    MagicNumber = 123456;          // Magic number
input int    SocketTimeout = 1000;          // Socket timeout (ms)

//--- Global variables
int socket_handle = INVALID_HANDLE;
CTrade trade;
datetime last_candle_time = 0;

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
      Print("Failed to connect to Natron server");
      return(INIT_FAILED);
   }
   
   Print("Natron EA initialized successfully");
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
   // Check for new candle
   datetime current_candle_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_candle_time != last_candle_time)
   {
      last_candle_time = current_candle_time;
      SendCandleData();
   }
   
   // Check for incoming predictions
   CheckForPredictions();
   
   // Update positions
   UpdatePositions();
}

//+------------------------------------------------------------------+
//| Connect to Python server                                         |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return false;
   }
   
   if(!SocketConnect(socket_handle, ServerHost, ServerPort, SocketTimeout))
   {
      Print("Failed to connect to server: ", ServerHost, ":", ServerPort);
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   Print("Connected to Natron server");
   return true;
}

//+------------------------------------------------------------------+
//| Send candle data to Python server                                |
//+------------------------------------------------------------------+
void SendCandleData()
{
   if(socket_handle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return;
   }
   
   // Get current candle data
   double open = iOpen(_Symbol, PERIOD_CURRENT, 0);
   double high = iHigh(_Symbol, PERIOD_CURRENT, 0);
   double low = iLow(_Symbol, PERIOD_CURRENT, 0);
   double close = iClose(_Symbol, PERIOD_CURRENT, 0);
   long volume = iVolume(_Symbol, PERIOD_CURRENT, 0);
   datetime time = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   // Create JSON message
   string json = "{"
                 "\"type\":\"candle\","
                 "\"data\":{"
                 "\"time\":" + IntegerToString(time) + ","
                 "\"open\":" + DoubleToString(open, _Digits) + ","
                 "\"high\":" + DoubleToString(high, _Digits) + ","
                 "\"low\":" + DoubleToString(low, _Digits) + ","
                 "\"close\":" + DoubleToString(close, _Digits) + ","
                 "\"volume\":" + IntegerToString(volume) + ","
                 "\"symbol\":\"" + _Symbol + "\""
                 "}"
                 "}";
   
   // Send to server
   uchar data[];
   StringToCharArray(json, data);
   if(SocketSend(socket_handle, data) <= 0)
   {
      Print("Failed to send candle data");
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
//| Check for predictions from Python server                         |
//+------------------------------------------------------------------+
void CheckForPredictions()
{
   if(socket_handle == INVALID_HANDLE)
      return;
   
   uchar data[];
   int received = SocketRead(socket_handle, data, SocketTimeout);
   
   if(received > 0)
   {
      string message = CharArrayToString(data);
      ProcessPrediction(message);
   }
}

//+------------------------------------------------------------------+
//| Process prediction from server                                   |
//+------------------------------------------------------------------+
void ProcessPrediction(string json_message)
{
   // Parse JSON (simplified - in production use proper JSON parser)
   // For now, extract key values
   int forecast_pos = StringFind(json_message, "\"forecast\":");
   int regime_pos = StringFind(json_message, "\"regime\":");
   int context_pos = StringFind(json_message, "\"context_strength\":");
   
   if(forecast_pos < 0 || regime_pos < 0)
      return;
   
   // Extract values (simplified parsing)
   // In production, use proper JSON parsing library
   
   // Execute trading logic based on prediction
   // This is a simplified version - full implementation would parse
   // all prediction fields and execute appropriate trades
}

//+------------------------------------------------------------------+
//| Update positions based on server commands                        |
//+------------------------------------------------------------------+
void UpdatePositions()
{
   // Check for open positions
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            // Update stop loss / take profit based on server commands
            // Implementation depends on server communication protocol
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Execute order from server                                        |
//+------------------------------------------------------------------+
void ExecuteOrder(string order_json)
{
   // Parse order JSON and execute trade
   // This would parse the order details and call trade.Buy() or trade.Sell()
   // Implementation depends on exact order format from server
}
