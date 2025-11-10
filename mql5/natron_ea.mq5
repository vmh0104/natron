//+------------------------------------------------------------------+
//|                                                     NatronEA.mq5 |
//|                                  Natron AI Trading System        |
//|                                  MetaTrader 5 Expert Advisor     |
//+------------------------------------------------------------------+
#property copyright "Natron AI System"
#property link      "https://github.com/natron-ai"
#property version   "2.00"
#property description "AI-powered trading using Natron Transformer model"
#property strict

//--- Input parameters
input string    ServerIP = "127.0.0.1";          // Python server IP
input int       ServerPort = 9090;               // Python server port
input int       SequenceLength = 96;             // Number of candles for prediction
input double    LotSize = 0.01;                  // Trading lot size
input double    MinBuyProb = 0.60;               // Minimum buy probability
input double    MinSellProb = 0.60;              // Minimum sell probability
input double    MinConfidence = 0.50;            // Minimum confidence
input int       StopLoss = 100;                  // Stop Loss in points
input int       TakeProfit = 200;                // Take Profit in points
input int       MaxSpread = 30;                  // Maximum spread in points
input bool      EnableTrading = true;            // Enable automated trading
input bool      ShowPanel = true;                // Show information panel
input int       UpdateInterval = 60;             // Update interval in seconds
input string    TradingHours = "00:00-23:59";    // Trading hours
input int       MaxPositions = 1;                // Maximum open positions
input bool      UseTrailingStop = false;         // Use trailing stop
input int       TrailingStop = 50;               // Trailing stop distance

//--- Global variables
int socket = INVALID_HANDLE;
datetime last_update_time = 0;
datetime last_bar_time = 0;
bool connected = false;

// Prediction results
double buy_prob = 0.0;
double sell_prob = 0.0;
double direction_up = 0.0;
string regime = "UNKNOWN";
double confidence = 0.0;
string signal = "HOLD";

// Trading statistics
int total_trades = 0;
int winning_trades = 0;
double total_profit = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=================================================================");
   Print("             NATRON AI TRADING SYSTEM - INITIALIZING            ");
   Print("=================================================================");
   
   // Create socket connection
   if(!ConnectToServer())
   {
      Print("❌ Failed to connect to Natron server");
      return(INIT_FAILED);
   }
   
   // Set up chart
   if(ShowPanel)
   {
      CreateInfoPanel();
   }
   
   Print("✅ Natron EA initialized successfully");
   Print("   Server: ", ServerIP, ":", ServerPort);
   Print("   Symbol: ", _Symbol);
   Print("   Timeframe: ", PeriodToString());
   Print("   Lot Size: ", LotSize);
   Print("   Trading: ", EnableTrading ? "ENABLED" : "DISABLED");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Close socket connection
   if(socket != INVALID_HANDLE)
   {
      SocketClose(socket);
      socket = INVALID_HANDLE;
   }
   
   // Clean up panel
   if(ShowPanel)
   {
      DeleteInfoPanel();
   }
   
   Print("Natron EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if new bar
   datetime current_bar_time = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(current_bar_time == last_bar_time)
      return;
   
   last_bar_time = current_bar_time;
   
   // Check update interval
   if(TimeCurrent() - last_update_time < UpdateInterval)
      return;
   
   last_update_time = TimeCurrent();
   
   // Check trading hours
   if(!IsInTradingHours())
      return;
   
   // Check spread
   int spread = (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(spread > MaxSpread)
   {
      Print("⚠️ Spread too high: ", spread, " > ", MaxSpread);
      return;
   }
   
   // Get prediction from Natron
   if(GetPrediction())
   {
      // Update panel
      if(ShowPanel)
         UpdateInfoPanel();
      
      // Execute trading logic
      if(EnableTrading)
         ExecuteTradingLogic();
   }
}

//+------------------------------------------------------------------+
//| Connect to Python server                                           |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   Print("🔌 Connecting to Natron server at ", ServerIP, ":", ServerPort);
   
   socket = SocketCreate();
   
   if(socket == INVALID_HANDLE)
   {
      Print("❌ Failed to create socket: ", GetLastError());
      return false;
   }
   
   if(!SocketConnect(socket, ServerIP, ServerPort, 5000))
   {
      Print("❌ Failed to connect: ", GetLastError());
      SocketClose(socket);
      socket = INVALID_HANDLE;
      return false;
   }
   
   connected = true;
   Print("✅ Connected to Natron server");
   return true;
}

//+------------------------------------------------------------------+
//| Get prediction from Natron server                                 |
//+------------------------------------------------------------------+
bool GetPrediction()
{
   if(!connected || socket == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return false;
   }
   
   // Prepare OHLCV data (last 96 candles)
   string json_data = PrepareOHLCVData();
   
   // Send request
   string request = json_data + "\n";
   char send_data[];
   StringToCharArray(request, send_data);
   ArrayResize(send_data, ArraySize(send_data) - 1); // Remove null terminator
   
   int sent = SocketSend(socket, send_data, ArraySize(send_data));
   if(sent <= 0)
   {
      Print("❌ Failed to send data: ", GetLastError());
      connected = false;
      return false;
   }
   
   // Receive response
   char recv_data[];
   string response = "";
   uint timeout_ms = 10000; // 10 seconds timeout
   
   uint len = SocketIsReadable(socket);
   if(len > 0)
   {
      ArrayResize(recv_data, len);
      int received = SocketRead(socket, recv_data, len, timeout_ms);
      
      if(received > 0)
      {
         response = CharArrayToString(recv_data, 0, received);
         
         // Parse JSON response
         if(ParsePrediction(response))
         {
            Print("📊 Prediction received: ", signal, " (confidence: ", DoubleToString(confidence, 2), ")");
            return true;
         }
      }
   }
   
   Print("❌ Failed to receive prediction");
   return false;
}

//+------------------------------------------------------------------+
//| Prepare OHLCV data as JSON                                        |
//+------------------------------------------------------------------+
string PrepareOHLCVData()
{
   string json = "{\"data\":[";
   
   for(int i = SequenceLength - 1; i >= 0; i--)
   {
      datetime time = iTime(_Symbol, PERIOD_CURRENT, i);
      double open = iOpen(_Symbol, PERIOD_CURRENT, i);
      double high = iHigh(_Symbol, PERIOD_CURRENT, i);
      double low = iLow(_Symbol, PERIOD_CURRENT, i);
      double close = iClose(_Symbol, PERIOD_CURRENT, i);
      long volume = iVolume(_Symbol, PERIOD_CURRENT, i);
      
      json += "{";
      json += "\"time\":\"" + TimeToString(time, TIME_DATE|TIME_MINUTES) + "\",";
      json += "\"open\":" + DoubleToString(open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString(volume);
      json += "}";
      
      if(i > 0)
         json += ",";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Parse prediction JSON response                                     |
//+------------------------------------------------------------------+
bool ParsePrediction(string response)
{
   // Simple JSON parsing (for production, use a proper JSON library)
   int pos;
   
   // Extract buy_prob
   pos = StringFind(response, "\"buy_prob\":");
   if(pos >= 0)
      buy_prob = StringToDouble(StringSubstr(response, pos + 11, 6));
   
   // Extract sell_prob
   pos = StringFind(response, "\"sell_prob\":");
   if(pos >= 0)
      sell_prob = StringToDouble(StringSubstr(response, pos + 12, 6));
   
   // Extract direction_up
   pos = StringFind(response, "\"direction_up\":");
   if(pos >= 0)
      direction_up = StringToDouble(StringSubstr(response, pos + 15, 6));
   
   // Extract confidence
   pos = StringFind(response, "\"confidence\":");
   if(pos >= 0)
      confidence = StringToDouble(StringSubstr(response, pos + 13, 6));
   
   // Extract regime
   pos = StringFind(response, "\"regime\":\"");
   if(pos >= 0)
   {
      int end_pos = StringFind(response, "\"", pos + 10);
      regime = StringSubstr(response, pos + 10, end_pos - pos - 10);
   }
   
   // Extract signal
   pos = StringFind(response, "\"signal\":\"");
   if(pos >= 0)
   {
      int end_pos = StringFind(response, "\"", pos + 10);
      signal = StringSubstr(response, pos + 10, end_pos - pos - 10);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic                                              |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
   // Count current positions
   int positions = CountPositions();
   
   if(positions >= MaxPositions)
   {
      Print("⚠️ Maximum positions reached: ", positions);
      return;
   }
   
   // Check confidence
   if(confidence < MinConfidence)
   {
      Print("⚠️ Confidence too low: ", confidence);
      return;
   }
   
   // Execute based on signal
   if(signal == "BUY" && buy_prob >= MinBuyProb)
   {
      if(positions == 0)
         OpenPosition(ORDER_TYPE_BUY);
   }
   else if(signal == "SELL" && sell_prob >= MinSellProb)
   {
      if(positions == 0)
         OpenPosition(ORDER_TYPE_SELL);
   }
   
   // Update trailing stops
   if(UseTrailingStop)
      UpdateTrailingStops();
}

//+------------------------------------------------------------------+
//| Open position                                                      |
//+------------------------------------------------------------------+
bool OpenPosition(ENUM_ORDER_TYPE type)
{
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   // Calculate SL and TP
   double sl = 0, tp = 0;
   if(StopLoss > 0)
   {
      sl = (type == ORDER_TYPE_BUY) ? price - StopLoss * point : price + StopLoss * point;
   }
   if(TakeProfit > 0)
   {
      tp = (type == ORDER_TYPE_BUY) ? price + TakeProfit * point : price - TakeProfit * point;
   }
   
   // Prepare request
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = type;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = 123456;
   request.comment = "Natron AI - " + signal + " (conf:" + DoubleToString(confidence, 2) + ")";
   
   // Send order
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✅ ", (type == ORDER_TYPE_BUY ? "BUY" : "SELL"), " order opened successfully");
         Print("   Ticket: ", result.deal);
         Print("   Price: ", price);
         Print("   SL: ", sl);
         Print("   TP: ", tp);
         total_trades++;
         return true;
      }
   }
   
   Print("❌ Failed to open position: ", result.retcode, " - ", result.comment);
   return false;
}

//+------------------------------------------------------------------+
//| Count open positions                                               |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetSymbol(i) == _Symbol)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Update trailing stops                                              |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double sl = PositionGetDouble(POSITION_SL);
         ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
         
         double current_price = (type == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         
         double new_sl = 0;
         
         if(type == POSITION_TYPE_BUY)
         {
            new_sl = current_price - TrailingStop * point;
            if(new_sl > sl && new_sl < current_price)
            {
               ModifyPosition(ticket, new_sl, PositionGetDouble(POSITION_TP));
            }
         }
         else if(type == POSITION_TYPE_SELL)
         {
            new_sl = current_price + TrailingStop * point;
            if((new_sl < sl || sl == 0) && new_sl > current_price)
            {
               ModifyPosition(ticket, new_sl, PositionGetDouble(POSITION_TP));
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Modify position                                                    |
//+------------------------------------------------------------------+
bool ModifyPosition(ulong ticket, double sl, double tp)
{
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   
   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl = sl;
   request.tp = tp;
   
   return OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Check if in trading hours                                          |
//+------------------------------------------------------------------+
bool IsInTradingHours()
{
   // Simple implementation - can be enhanced
   return true;
}

//+------------------------------------------------------------------+
//| Period to string                                                   |
//+------------------------------------------------------------------+
string PeriodToString()
{
   switch(Period())
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      default:         return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| Create information panel                                           |
//+------------------------------------------------------------------+
void CreateInfoPanel()
{
   // Create panel objects
   ObjectCreate(0, "NatronPanel", OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_YDISTANCE, 30);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_XSIZE, 300);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_YSIZE, 250);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_BGCOLOR, clrNavy);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, "NatronPanel", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   
   // Title
   ObjectCreate(0, "NatronTitle", OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, "NatronTitle", OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, "NatronTitle", OBJPROP_XDISTANCE, 20);
   ObjectSetInteger(0, "NatronTitle", OBJPROP_YDISTANCE, 40);
   ObjectSetString(0, "NatronTitle", OBJPROP_TEXT, "🧠 NATRON AI TRADER");
   ObjectSetInteger(0, "NatronTitle", OBJPROP_FONTSIZE, 12);
   ObjectSetInteger(0, "NatronTitle", OBJPROP_COLOR, clrWhite);
}

//+------------------------------------------------------------------+
//| Update information panel                                           |
//+------------------------------------------------------------------+
void UpdateInfoPanel()
{
   // Update signal
   string label_signal = "NatronSignal";
   if(ObjectFind(0, label_signal) < 0)
   {
      ObjectCreate(0, label_signal, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, label_signal, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, label_signal, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, label_signal, OBJPROP_YDISTANCE, 70);
      ObjectSetInteger(0, label_signal, OBJPROP_FONTSIZE, 10);
   }
   ObjectSetString(0, label_signal, OBJPROP_TEXT, "Signal: " + signal);
   color signal_color = (signal == "BUY") ? clrLime : (signal == "SELL") ? clrRed : clrYellow;
   ObjectSetInteger(0, label_signal, OBJPROP_COLOR, signal_color);
   
   // Update other info...
   CreateOrUpdateLabel("NatronBuyProb", 20, 95, "Buy Prob: " + DoubleToString(buy_prob, 2));
   CreateOrUpdateLabel("NatronSellProb", 20, 115, "Sell Prob: " + DoubleToString(sell_prob, 2));
   CreateOrUpdateLabel("NatronDirection", 20, 135, "Up Prob: " + DoubleToString(direction_up, 2));
   CreateOrUpdateLabel("NatronRegime", 20, 155, "Regime: " + regime);
   CreateOrUpdateLabel("NatronConfidence", 20, 175, "Confidence: " + DoubleToString(confidence, 2));
   CreateOrUpdateLabel("NatronTrades", 20, 200, "Trades: " + IntegerToString(total_trades));
}

//+------------------------------------------------------------------+
//| Helper to create or update label                                  |
//+------------------------------------------------------------------+
void CreateOrUpdateLabel(string name, int x, int y, string text)
{
   if(ObjectFind(0, name) < 0)
   {
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clrWhite);
   }
   ObjectSetString(0, name, OBJPROP_TEXT, text);
}

//+------------------------------------------------------------------+
//| Delete information panel                                           |
//+------------------------------------------------------------------+
void DeleteInfoPanel()
{
   ObjectDelete(0, "NatronPanel");
   ObjectDelete(0, "NatronTitle");
   ObjectDelete(0, "NatronSignal");
   ObjectDelete(0, "NatronBuyProb");
   ObjectDelete(0, "NatronSellProb");
   ObjectDelete(0, "NatronDirection");
   ObjectDelete(0, "NatronRegime");
   ObjectDelete(0, "NatronConfidence");
   ObjectDelete(0, "NatronTrades");
}
//+------------------------------------------------------------------+
