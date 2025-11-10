//+------------------------------------------------------------------+
//|                                            natron_ea.mq5         |
//|                        Natron AI Trading System - MQL5 EA         |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input double   InpLotSize = 0.01;           // Lot size
input int      InpMagicNumber = 123456;     // Magic number
input int      InpSocketPort = 8888;        // Socket port for Python server
input string   InpSymbol = "EURUSD";         // Trading symbol
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M15; // Timeframe

//--- Global variables
CTrade trade;
int socket_handle = INVALID_HANDLE;
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set magic number
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Initialize socket connection to Python server
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return(INIT_FAILED);
   }
   
   // Connect to Python server (localhost)
   if(!SocketConnect(socket_handle, "localhost", InpSocketPort, 1000))
   {
      Print("Failed to connect to Python server on port ", InpSocketPort);
      SocketClose(socket_handle);
      return(INIT_FAILED);
   }
   
   Print("Natron EA initialized successfully");
   Print("Connected to Python server on port ", InpSocketPort);
   
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
   // Check for new bar
   datetime current_bar_time = iTime(InpSymbol, InpTimeframe, 0);
   if(current_bar_time != last_bar_time)
   {
      last_bar_time = current_bar_time;
      
      // Send candle data to Python server
      SendCandleData();
      
      // Receive and process signals from Python server
      ProcessSignals();
   }
}

//+------------------------------------------------------------------+
//| Send candle data to Python server                                |
//+------------------------------------------------------------------+
void SendCandleData()
{
   if(socket_handle == INVALID_HANDLE)
      return;
   
   // Get OHLCV data
   double open = iOpen(InpSymbol, InpTimeframe, 0);
   double high = iHigh(InpSymbol, InpTimeframe, 0);
   double low = iLow(InpSymbol, InpTimeframe, 0);
   double close = iClose(InpSymbol, InpTimeframe, 0);
   long volume = iVolume(InpSymbol, InpTimeframe, 0);
   datetime time = iTime(InpSymbol, InpTimeframe, 0);
   
   // Create JSON message
   string json_msg = StringFormat(
      "{\"time\":%d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%lld}\n",
      (int)time, open, high, low, close, volume
   );
   
   // Send to Python server
   uchar data[];
   StringToCharArray(json_msg, data, 0, StringLen(json_msg));
   SocketSend(socket_handle, data);
}

//+------------------------------------------------------------------+
//| Process trading signals from Python server                       |
//+------------------------------------------------------------------+
void ProcessSignals()
{
   if(socket_handle == INVALID_HANDLE)
      return;
   
   // Receive data from Python server
   uchar data[];
   int received = SocketRead(socket_handle, data, 0, 1000);
   
   if(received > 0)
   {
      string message = CharArrayToString(data, 0, received);
      
      // Parse JSON (simplified - in production use proper JSON parser)
      if(StringFind(message, "\"direction\":\"BUY\"") >= 0)
      {
         // Extract entry parameters from JSON
         double entry_price = ExtractDouble(message, "entry_price");
         double stop_loss = ExtractDouble(message, "stop_loss");
         double take_profit = ExtractDouble(message, "take_profit");
         
         // Execute BUY order
         if(entry_price > 0 && stop_loss > 0 && take_profit > 0)
         {
            ExecuteBuyOrder(entry_price, stop_loss, take_profit);
         }
      }
      else if(StringFind(message, "\"direction\":\"SELL\"") >= 0)
      {
         // Extract entry parameters from JSON
         double entry_price = ExtractDouble(message, "entry_price");
         double stop_loss = ExtractDouble(message, "stop_loss");
         double take_profit = ExtractDouble(message, "take_profit");
         
         // Execute SELL order
         if(entry_price > 0 && stop_loss > 0 && take_profit > 0)
         {
            ExecuteSellOrder(entry_price, stop_loss, take_profit);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Extract double value from JSON string                            |
//+------------------------------------------------------------------+
double ExtractDouble(string json, string key)
{
   string search_key = "\"" + key + "\":";
   int pos = StringFind(json, search_key);
   if(pos < 0)
      return 0.0;
   
   int start = pos + StringLen(search_key);
   int end = StringFind(json, ",", start);
   if(end < 0)
      end = StringFind(json, "}", start);
   if(end < 0)
      return 0.0;
   
   string value_str = StringSubstr(json, start, end - start);
   StringTrimLeft(value_str);
   StringTrimRight(value_str);
   
   return StringToDouble(value_str);
}

//+------------------------------------------------------------------+
//| Execute BUY order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder(double entry_price, double stop_loss, double take_profit)
{
   // Check if we already have a position
   if(PositionSelect(InpSymbol))
   {
      if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
      {
         Print("Position already exists, skipping new order");
         return;
      }
   }
   
   // Normalize prices
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   entry_price = NormalizeDouble(entry_price, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
   stop_loss = NormalizeDouble(stop_loss, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
   take_profit = NormalizeDouble(take_profit, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
   
   // Calculate SL/TP in points
   long sl_points = (long)MathAbs((entry_price - stop_loss) / point);
   long tp_points = (long)MathAbs((take_profit - entry_price) / point);
   
   // Execute BUY limit order
   if(trade.BuyLimit(InpLotSize, entry_price, InpSymbol, 0, stop_loss, take_profit, "Natron AI Signal"))
   {
      Print("BUY LIMIT order placed: Entry=", entry_price, " SL=", stop_loss, " TP=", take_profit);
   }
   else
   {
      Print("Failed to place BUY LIMIT order: ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Execute SELL order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder(double entry_price, double stop_loss, double take_profit)
{
   // Check if we already have a position
   if(PositionSelect(InpSymbol))
   {
      if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
      {
         Print("Position already exists, skipping new order");
         return;
      }
   }
   
   // Normalize prices
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   entry_price = NormalizeDouble(entry_price, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
   stop_loss = NormalizeDouble(stop_loss, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
   take_profit = NormalizeDouble(take_profit, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
   
   // Calculate SL/TP in points
   long sl_points = (long)MathAbs((stop_loss - entry_price) / point);
   long tp_points = (long)MathAbs((entry_price - take_profit) / point);
   
   // Execute SELL limit order
   if(trade.SellLimit(InpLotSize, entry_price, InpSymbol, 0, stop_loss, take_profit, "Natron AI Signal"))
   {
      Print("SELL LIMIT order placed: Entry=", entry_price, " SL=", stop_loss, " TP=", take_profit);
   }
   else
   {
      Print("Failed to place SELL LIMIT order: ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Trailing stop function                                           |
//+------------------------------------------------------------------+
void UpdateTrailingStop()
{
   if(!PositionSelect(InpSymbol))
      return;
   
   if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
      return;
   
   long position_type = PositionGetInteger(POSITION_TYPE);
   double position_open_price = PositionGetDouble(POSITION_PRICE_OPEN);
   double current_sl = PositionGetDouble(POSITION_SL);
   double current_tp = PositionGetDouble(POSITION_TP);
   double current_price = (position_type == POSITION_TYPE_BUY) ? 
                          SymbolInfoDouble(InpSymbol, SYMBOL_BID) : 
                          SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
   
   // Calculate ATR for trailing stop
   int atr_handle = iATR(InpSymbol, InpTimeframe, 14);
   double atr[];
   ArraySetAsSeries(atr, true);
   CopyBuffer(atr_handle, 0, 0, 1, atr);
   IndicatorRelease(atr_handle);
   
   if(ArraySize(atr) == 0)
      return;
   
   double trailing_distance = atr[0] * 2.0; // 2x ATR trailing stop
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   
   if(position_type == POSITION_TYPE_BUY)
   {
      double new_sl = current_price - trailing_distance;
      new_sl = NormalizeDouble(new_sl, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
      
      if(new_sl > current_sl && new_sl < current_price)
      {
         trade.PositionModify(InpSymbol, new_sl, current_tp);
         Print("Trailing stop updated for BUY position: New SL=", new_sl);
      }
   }
   else if(position_type == POSITION_TYPE_SELL)
   {
      double new_sl = current_price + trailing_distance;
      new_sl = NormalizeDouble(new_sl, (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS));
      
      if(new_sl < current_sl && new_sl > current_price)
      {
         trade.PositionModify(InpSymbol, new_sl, current_tp);
         Print("Trailing stop updated for SELL position: New SL=", new_sl);
      }
   }
}
