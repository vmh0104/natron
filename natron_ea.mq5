//+------------------------------------------------------------------+
//|                                                    Natron_EA.mq5 |
//|                                  Natron AI Trading System        |
//|                                  MQL5 Expert Advisor             |
//+------------------------------------------------------------------+
#property copyright "Natron AI Team"
#property link      ""
#property version   "2.00"
#property description "AI-powered trading system with Transformer model"

//--- Input parameters
input string   ServerHost = "127.0.0.1";        // Python server IP
input int      ServerPort = 9999;               // Python server port
input int      SequenceLength = 96;             // Number of candles to send
input double   BuyThreshold = 0.65;             // Buy signal threshold
input double   SellThreshold = 0.65;            // Sell signal threshold
input double   LotSize = 0.1;                   // Position size
input int      StopLoss = 100;                  // Stop loss in points
input int      TakeProfit = 200;                // Take profit in points
input int      MaxOpenPositions = 1;            // Max simultaneous positions
input bool     UseTrailing = true;              // Use trailing stop
input int      TrailingStop = 50;               // Trailing stop in points
input int      PredictionInterval = 60;         // Seconds between predictions

//--- Global variables
int socket_handle = INVALID_HANDLE;
datetime last_prediction_time = 0;
double last_buy_prob = 0;
double last_sell_prob = 0;
string last_regime = "";
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("==========================================================");
   Print("🚀 Natron AI Expert Advisor V2 Initializing...");
   Print("==========================================================");
   Print("📡 Server: ", ServerHost, ":", ServerPort);
   Print("📊 Sequence Length: ", SequenceLength);
   Print("🎯 Buy Threshold: ", BuyThreshold);
   Print("🎯 Sell Threshold: ", SellThreshold);
   Print("💰 Lot Size: ", LotSize);
   Print("==========================================================");
   
   // Test connection
   if(!TestConnection())
   {
      Print("❌ Failed to connect to Natron server");
      return(INIT_FAILED);
   }
   
   Print("✅ Connection successful!");
   Print("✅ Natron EA initialized");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("🛑 Natron EA shutting down...");
   CloseSocket();
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if new bar
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   bool new_bar = (current_bar_time != last_bar_time);
   
   if(new_bar)
   {
      last_bar_time = current_bar_time;
   }
   
   // Check if it's time for new prediction
   datetime current_time = TimeCurrent();
   
   if(current_time - last_prediction_time >= PredictionInterval || new_bar)
   {
      // Get prediction from Natron AI
      if(GetPrediction())
      {
         last_prediction_time = current_time;
         
         // Display on chart
         DisplaySignals();
         
         // Execute trading logic
         ExecuteTrading();
      }
   }
   
   // Update trailing stops
   if(UseTrailing)
   {
      UpdateTrailingStops();
   }
}

//+------------------------------------------------------------------+
//| Test connection to Python server                                 |
//+------------------------------------------------------------------+
bool TestConnection()
{
   int handle = SocketCreate();
   
   if(handle == INVALID_HANDLE)
   {
      Print("❌ Failed to create socket: ", GetLastError());
      return false;
   }
   
   if(!SocketConnect(handle, ServerHost, ServerPort, 5000))
   {
      Print("❌ Failed to connect to server: ", GetLastError());
      SocketClose(handle);
      return false;
   }
   
   SocketClose(handle);
   return true;
}

//+------------------------------------------------------------------+
//| Get prediction from Natron server                                |
//+------------------------------------------------------------------+
bool GetPrediction()
{
   // Collect OHLCV data
   string json_data = CollectOHLCVData();
   
   if(json_data == "")
   {
      Print("❌ Failed to collect OHLCV data");
      return false;
   }
   
   // Send request to server
   string response = SendSocketRequest(json_data);
   
   if(response == "")
   {
      Print("❌ No response from server");
      return false;
   }
   
   // Parse response
   return ParsePrediction(response);
}

//+------------------------------------------------------------------+
//| Collect OHLCV data from chart                                    |
//+------------------------------------------------------------------+
string CollectOHLCVData()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(_Symbol, _Period, 0, SequenceLength, rates);
   
   if(copied < SequenceLength)
   {
      Print("❌ Failed to copy rates: ", GetLastError());
      return "";
   }
   
   // Build JSON
   string json = "{\"data\":[";
   
   for(int i = SequenceLength - 1; i >= 0; i--)
   {
      string time_str = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
      
      json += "{";
      json += "\"time\":\"" + time_str + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
      json += "}";
      
      if(i > 0) json += ",";
   }
   
   json += "]}";
   
   return json;
}

//+------------------------------------------------------------------+
//| Send socket request and get response                             |
//+------------------------------------------------------------------+
string SendSocketRequest(string request)
{
   // Create socket
   int handle = SocketCreate();
   if(handle == INVALID_HANDLE)
   {
      Print("❌ Socket creation failed: ", GetLastError());
      return "";
   }
   
   // Connect
   if(!SocketConnect(handle, ServerHost, ServerPort, 5000))
   {
      Print("❌ Connection failed: ", GetLastError());
      SocketClose(handle);
      return "";
   }
   
   // Send request
   char req[];
   StringToCharArray(request, req, 0, WHOLE_ARRAY, CP_UTF8);
   
   int sent = SocketSend(handle, req, ArraySize(req));
   if(sent <= 0)
   {
      Print("❌ Send failed: ", GetLastError());
      SocketClose(handle);
      return "";
   }
   
   // Receive response
   string response = "";
   char buffer[];
   ArrayResize(buffer, 4096);
   
   uint timeout = GetTickCount() + 5000;
   
   while(GetTickCount() < timeout)
   {
      int received = SocketRead(handle, buffer, ArraySize(buffer), 100);
      
      if(received > 0)
      {
         response += CharArrayToString(buffer, 0, received, CP_UTF8);
         
         // Check if complete JSON received
         if(StringFind(response, "}") >= 0)
            break;
      }
      else if(received < 0)
      {
         break;
      }
   }
   
   SocketClose(handle);
   
   return response;
}

//+------------------------------------------------------------------+
//| Parse prediction response                                         |
//+------------------------------------------------------------------+
bool ParsePrediction(string json)
{
   // Simple JSON parsing (in production, use proper JSON library)
   int buy_pos = StringFind(json, "\"buy_prob\":");
   int sell_pos = StringFind(json, "\"sell_prob\":");
   int regime_pos = StringFind(json, "\"regime\":\"");
   
   if(buy_pos < 0 || sell_pos < 0 || regime_pos < 0)
   {
      Print("❌ Invalid JSON response");
      return false;
   }
   
   // Extract buy_prob
   int start = buy_pos + 11;
   int end = StringFind(json, ",", start);
   string buy_str = StringSubstr(json, start, end - start);
   last_buy_prob = StringToDouble(buy_str);
   
   // Extract sell_prob
   start = sell_pos + 12;
   end = StringFind(json, ",", start);
   string sell_str = StringSubstr(json, start, end - start);
   last_sell_prob = StringToDouble(sell_str);
   
   // Extract regime
   start = regime_pos + 10;
   end = StringFind(json, "\"", start);
   last_regime = StringSubstr(json, start, end - start);
   
   Print("📊 Prediction: Buy=", DoubleToString(last_buy_prob, 2), 
         " | Sell=", DoubleToString(last_sell_prob, 2),
         " | Regime=", last_regime);
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic based on predictions                        |
//+------------------------------------------------------------------+
void ExecuteTrading()
{
   // Count open positions
   int open_positions = CountOpenPositions();
   
   if(open_positions >= MaxOpenPositions)
      return;
   
   // Buy signal
   if(last_buy_prob >= BuyThreshold && last_sell_prob < SellThreshold)
   {
      if(open_positions == 0 || !HasPosition(POSITION_TYPE_BUY))
      {
         OpenPosition(ORDER_TYPE_BUY);
      }
   }
   // Sell signal
   else if(last_sell_prob >= SellThreshold && last_buy_prob < BuyThreshold)
   {
      if(open_positions == 0 || !HasPosition(POSITION_TYPE_SELL))
      {
         OpenPosition(ORDER_TYPE_SELL);
      }
   }
}

//+------------------------------------------------------------------+
//| Open trading position                                             |
//+------------------------------------------------------------------+
bool OpenPosition(ENUM_ORDER_TYPE order_type)
{
   double price = (order_type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) 
                                                   : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   double sl = 0, tp = 0;
   
   if(StopLoss > 0)
   {
      if(order_type == ORDER_TYPE_BUY)
         sl = price - StopLoss * _Point;
      else
         sl = price + StopLoss * _Point;
   }
   
   if(TakeProfit > 0)
   {
      if(order_type == ORDER_TYPE_BUY)
         tp = price + TakeProfit * _Point;
      else
         tp = price - TakeProfit * _Point;
   }
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = order_type;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = 123456;
   request.comment = "Natron AI - " + last_regime;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         string direction = (order_type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
         Print("✅ ", direction, " order opened at ", price, " | Regime: ", last_regime);
         return true;
      }
      else
      {
         Print("❌ Order failed: ", result.retcode);
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Count open positions                                              |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int count = 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == 123456)
            count++;
      }
   }
   
   return count;
}

//+------------------------------------------------------------------+
//| Check if has position of type                                     |
//+------------------------------------------------------------------+
bool HasPosition(ENUM_POSITION_TYPE position_type)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == 123456)
         {
            if(PositionGetInteger(POSITION_TYPE) == position_type)
               return true;
         }
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Update trailing stops                                             |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         if(PositionGetInteger(POSITION_MAGIC) == 123456)
         {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            double current_sl = PositionGetDouble(POSITION_SL);
            double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
            ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
            
            double current_price = (pos_type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                                                     : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            
            double new_sl = 0;
            
            if(pos_type == POSITION_TYPE_BUY)
            {
               new_sl = current_price - TrailingStop * _Point;
               
               if(new_sl > current_sl && new_sl > open_price)
               {
                  ModifyPosition(ticket, new_sl, PositionGetDouble(POSITION_TP));
               }
            }
            else
            {
               new_sl = current_price + TrailingStop * _Point;
               
               if((current_sl == 0 || new_sl < current_sl) && new_sl < open_price)
               {
                  ModifyPosition(ticket, new_sl, PositionGetDouble(POSITION_TP));
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Modify position SL/TP                                             |
//+------------------------------------------------------------------+
bool ModifyPosition(ulong ticket, double sl, double tp)
{
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl = NormalizeDouble(sl, _Digits);
   request.tp = NormalizeDouble(tp, _Digits);
   
   return OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Display signals on chart                                          |
//+------------------------------------------------------------------+
void DisplaySignals()
{
   // Create info panel
   string panel_name = "Natron_Panel";
   
   int x = 20;
   int y = 50;
   int line_height = 20;
   
   // Background
   CreateLabel(panel_name + "_bg", x-5, y-5, "■■■■■■■■■■■■■■■", clrDarkSlateGray, 14);
   
   // Title
   CreateLabel(panel_name + "_title", x, y, "🧠 NATRON AI", clrLime, 12, true);
   y += line_height + 5;
   
   // Buy probability
   color buy_color = (last_buy_prob >= BuyThreshold) ? clrLime : clrGray;
   CreateLabel(panel_name + "_buy", x, y, "BUY:  " + DoubleToString(last_buy_prob * 100, 1) + "%", buy_color, 10);
   y += line_height;
   
   // Sell probability
   color sell_color = (last_sell_prob >= SellThreshold) ? clrRed : clrGray;
   CreateLabel(panel_name + "_sell", x, y, "SELL: " + DoubleToString(last_sell_prob * 100, 1) + "%", sell_color, 10);
   y += line_height;
   
   // Regime
   CreateLabel(panel_name + "_regime", x, y, "Regime: " + last_regime, clrYellow, 10);
   y += line_height;
   
   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Create label on chart                                             |
//+------------------------------------------------------------------+
void CreateLabel(string name, int x, int y, string text, color clr, int font_size, bool bold = false)
{
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   }
   
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, font_size);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   
   if(bold)
      ObjectSetString(0, name, OBJPROP_FONT, "Arial Bold");
   else
      ObjectSetString(0, name, OBJPROP_FONT, "Arial");
}

//+------------------------------------------------------------------+
//| Close socket                                                      |
//+------------------------------------------------------------------+
void CloseSocket()
{
   if(socket_handle != INVALID_HANDLE)
   {
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
   }
}
//+------------------------------------------------------------------+
