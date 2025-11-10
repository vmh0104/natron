//+------------------------------------------------------------------+
//| Natron Transformer Expert Advisor for MetaTrader 5              |
//| Real-time AI-powered trading using Natron model                 |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property link      "https://github.com/natron-ai"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\SymbolInfo.mqh>

//--- Input parameters
input string   ServerHost = "127.0.0.1";          // Python server host
input int      ServerPort = 9090;                  // Python server port
input int      SequenceLength = 96;                // Number of candles to send
input ENUM_TIMEFRAMES Timeframe = PERIOD_M15;     // Timeframe
input double   LotSize = 0.1;                      // Lot size
input int      Magic = 20251110;                   // Magic number
input double   ConfidenceThreshold = 0.6;          // Minimum confidence
input bool     EnableBuySignals = true;            // Enable BUY signals
input bool     EnableSellSignals = true;           // Enable SELL signals
input int      MaxSlippage = 10;                   // Max slippage in points
input double   StopLossPips = 50;                  // Stop Loss in pips
input double   TakeProfitPips = 100;               // Take Profit in pips
input bool     TrailStop = true;                   // Enable trailing stop
input double   TrailStopPips = 30;                 // Trailing stop distance
input int      RequestInterval = 60;               // Request interval in seconds

//--- Global variables
CTrade trade;
CPositionInfo position;
CSymbolInfo symbol;
int socket_handle = INVALID_HANDLE;
datetime last_request_time = 0;
string last_signal = "HOLD";
double last_confidence = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Set magic number
   trade.SetExpertMagicNumber(Magic);
   trade.SetMarginMode();
   trade.SetTypeFillingBySymbol(Symbol());
   trade.SetDeviationInPoints(MaxSlippage);
   
   //--- Initialize symbol
   if(!symbol.Name(_Symbol))
   {
      symbol.Name(_Symbol);
      symbol.Refresh();
   }
   
   //--- Test socket connection
   Print("🚀 Natron EA initialized");
   Print("   Symbol: ", Symbol());
   Print("   Timeframe: ", EnumToString(Timeframe));
   Print("   Server: ", ServerHost, ":", ServerPort);
   Print("   Sequence Length: ", SequenceLength);
   Print("   Lot Size: ", LotSize);
   
   //--- Connect to server
   if(!ConnectToServer())
   {
      Print("⚠️  Failed to connect to Python server");
      Print("   Make sure socket_server.py is running");
      return(INIT_FAILED);
   }
   
   Print("✅ Connected to Natron AI server");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Close socket
   if(socket_handle != INVALID_HANDLE)
   {
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
   }
   
   Print("🛑 Natron EA stopped: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check if new bar
   static datetime last_bar_time = 0;
   datetime current_bar_time = iTime(Symbol(), Timeframe, 0);
   
   if(current_bar_time == last_bar_time)
      return; // No new bar
   
   last_bar_time = current_bar_time;
   
   //--- Check request interval
   if(TimeCurrent() - last_request_time < RequestInterval)
      return;
   
   //--- Get prediction from AI
   string prediction_json = "";
   if(!GetAIPrediction(prediction_json))
   {
      Print("❌ Failed to get AI prediction");
      return;
   }
   
   //--- Parse prediction
   string signal;
   double buy_prob, sell_prob, confidence;
   string regime;
   
   if(!ParsePrediction(prediction_json, signal, buy_prob, sell_prob, confidence, regime))
   {
      Print("❌ Failed to parse prediction");
      return;
   }
   
   //--- Update last values
   last_signal = signal;
   last_confidence = confidence;
   last_request_time = TimeCurrent();
   
   //--- Display on chart
   DisplaySignalOnChart(signal, buy_prob, sell_prob, confidence, regime);
   
   //--- Execute trading logic
   if(confidence >= ConfidenceThreshold)
   {
      ExecuteSignal(signal, buy_prob, sell_prob, confidence);
   }
   
   //--- Manage open positions
   ManagePositions();
}

//+------------------------------------------------------------------+
//| Connect to Python socket server                                  |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   //--- Create socket
   socket_handle = SocketCreate();
   
   if(socket_handle == INVALID_HANDLE)
   {
      Print("Failed to create socket: ", GetLastError());
      return false;
   }
   
   //--- Connect
   if(!SocketConnect(socket_handle, ServerHost, ServerPort, 5000))
   {
      Print("Failed to connect: ", GetLastError());
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   //--- Send ping
   string ping = "{\"type\":\"ping\"}";
   if(!SendRequest(ping))
   {
      Print("Failed to send ping");
      return false;
   }
   
   string response = "";
   if(!ReceiveResponse(response))
   {
      Print("Failed to receive pong");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Get AI prediction from server                                    |
//+------------------------------------------------------------------+
bool GetAIPrediction(string &response)
{
   //--- Collect OHLCV data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates(Symbol(), Timeframe, 0, SequenceLength, rates);
   
   if(copied < SequenceLength)
   {
      Print("Failed to copy rates: ", GetLastError());
      return false;
   }
   
   //--- Build JSON request
   string request = "{\"type\":\"predict\",\"data\":[";
   
   for(int i = SequenceLength - 1; i >= 0; i--)
   {
      string candle = StringFormat(
         "{\"time\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%d}",
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES),
         rates[i].open,
         rates[i].high,
         rates[i].low,
         rates[i].close,
         rates[i].tick_volume
      );
      
      request += candle;
      if(i > 0) request += ",";
   }
   
   request += "]}";
   
   //--- Send request
   if(!SendRequest(request))
      return false;
   
   //--- Receive response
   if(!ReceiveResponse(response))
      return false;
   
   return true;
}

//+------------------------------------------------------------------+
//| Send request to server                                           |
//+------------------------------------------------------------------+
bool SendRequest(string request)
{
   //--- Reconnect if needed
   if(socket_handle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return false;
   }
   
   //--- Send
   char data[];
   StringToCharArray(request, data, 0, StringLen(request));
   
   int sent = SocketSend(socket_handle, data, ArraySize(data));
   
   if(sent <= 0)
   {
      Print("Failed to send data: ", GetLastError());
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Receive response from server                                     |
//+------------------------------------------------------------------+
bool ReceiveResponse(string &response)
{
   char buffer[];
   ArrayResize(buffer, 4096);
   
   uint timeout = 5000; // 5 seconds
   int received = SocketReceive(socket_handle, buffer, ArraySize(buffer), timeout);
   
   if(received <= 0)
   {
      Print("Failed to receive data: ", GetLastError());
      return false;
   }
   
   response = CharArrayToString(buffer, 0, received);
   return true;
}

//+------------------------------------------------------------------+
//| Parse prediction response                                        |
//+------------------------------------------------------------------+
bool ParsePrediction(string json, string &signal, double &buy_prob, 
                     double &sell_prob, double &confidence, string &regime)
{
   //--- Simple JSON parsing (in production, use proper JSON library)
   //--- Extract values using string functions
   
   int signal_pos = StringFind(json, "\"signal\"");
   if(signal_pos >= 0)
   {
      int start = StringFind(json, "\"", signal_pos + 10) + 1;
      int end = StringFind(json, "\"", start);
      signal = StringSubstr(json, start, end - start);
   }
   
   //--- Extract buy_prob
   ExtractDoubleValue(json, "\"buy_prob\"", buy_prob);
   
   //--- Extract sell_prob
   ExtractDoubleValue(json, "\"sell_prob\"", sell_prob);
   
   //--- Extract confidence
   ExtractDoubleValue(json, "\"confidence\"", confidence);
   
   //--- Extract regime
   int regime_pos = StringFind(json, "\"regime\"");
   if(regime_pos >= 0)
   {
      int start = StringFind(json, "\"", regime_pos + 10) + 1;
      int end = StringFind(json, "\"", start);
      regime = StringSubstr(json, start, end - start);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                   |
//+------------------------------------------------------------------+
void ExtractDoubleValue(string json, string key, double &value)
{
   int pos = StringFind(json, key);
   if(pos >= 0)
   {
      int start = StringFind(json, ":", pos) + 1;
      int end = start;
      
      while(end < StringLen(json))
      {
         string ch = StringSubstr(json, end, 1);
         if(ch == "," || ch == "}" || ch == "]")
            break;
         end++;
      }
      
      string val_str = StringSubstr(json, start, end - start);
      StringTrimLeft(val_str);
      StringTrimRight(val_str);
      value = StringToDouble(val_str);
   }
}

//+------------------------------------------------------------------+
//| Execute trading signal                                           |
//+------------------------------------------------------------------+
void ExecuteSignal(string signal, double buy_prob, double sell_prob, double confidence)
{
   //--- Check if already have position
   if(position.Select(Symbol()))
   {
      //--- Position exists, check if should close
      if(signal == "SELL" && position.PositionType() == POSITION_TYPE_BUY)
      {
         trade.PositionClose(Symbol());
         Print("🔄 Closed BUY position (AI signal: SELL)");
      }
      else if(signal == "BUY" && position.PositionType() == POSITION_TYPE_SELL)
      {
         trade.PositionClose(Symbol());
         Print("🔄 Closed SELL position (AI signal: BUY)");
      }
      return;
   }
   
   //--- No position, check if should open
   double price = symbol.Ask();
   double sl = 0, tp = 0;
   
   if(signal == "BUY" && EnableBuySignals)
   {
      //--- Calculate SL/TP
      if(StopLossPips > 0)
         sl = price - StopLossPips * symbol.Point() * 10;
      if(TakeProfitPips > 0)
         tp = price + TakeProfitPips * symbol.Point() * 10;
      
      //--- Open BUY
      if(trade.Buy(LotSize, Symbol(), price, sl, tp, "Natron AI BUY"))
      {
         Print("✅ BUY order opened");
         Print("   Price: ", price);
         Print("   SL: ", sl);
         Print("   TP: ", tp);
         Print("   Confidence: ", confidence);
      }
   }
   else if(signal == "SELL" && EnableSellSignals)
   {
      price = symbol.Bid();
      
      //--- Calculate SL/TP
      if(StopLossPips > 0)
         sl = price + StopLossPips * symbol.Point() * 10;
      if(TakeProfitPips > 0)
         tp = price - TakeProfitPips * symbol.Point() * 10;
      
      //--- Open SELL
      if(trade.Sell(LotSize, Symbol(), price, sl, tp, "Natron AI SELL"))
      {
         Print("✅ SELL order opened");
         Print("   Price: ", price);
         Print("   SL: ", sl);
         Print("   TP: ", tp);
         Print("   Confidence: ", confidence);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage open positions (trailing stop, etc.)                     |
//+------------------------------------------------------------------+
void ManagePositions()
{
   if(!position.Select(Symbol()))
      return;
   
   if(!TrailStop)
      return;
   
   double current_price;
   double sl = position.StopLoss();
   double new_sl = 0;
   
   if(position.PositionType() == POSITION_TYPE_BUY)
   {
      current_price = symbol.Bid();
      double trail_level = current_price - TrailStopPips * symbol.Point() * 10;
      
      if(trail_level > sl && trail_level < current_price)
      {
         new_sl = trail_level;
      }
   }
   else if(position.PositionType() == POSITION_TYPE_SELL)
   {
      current_price = symbol.Ask();
      double trail_level = current_price + TrailStopPips * symbol.Point() * 10;
      
      if((sl == 0 || trail_level < sl) && trail_level > current_price)
      {
         new_sl = trail_level;
      }
   }
   
   //--- Modify SL if needed
   if(new_sl > 0)
   {
      if(trade.PositionModify(Symbol(), new_sl, position.TakeProfit()))
      {
         Print("🔄 Trailing stop updated to ", new_sl);
      }
   }
}

//+------------------------------------------------------------------+
//| Display signal on chart                                          |
//+------------------------------------------------------------------+
void DisplaySignalOnChart(string signal, double buy_prob, double sell_prob, 
                          double confidence, string regime)
{
   //--- Create info panel
   string label_name = "NatronAI_Panel";
   
   int x = 20;
   int y = 50;
   
   color signal_color = clrYellow;
   if(signal == "BUY") signal_color = clrLimeGreen;
   else if(signal == "SELL") signal_color = clrRed;
   
   string info = StringFormat(
      "🧠 NATRON AI\n" +
      "━━━━━━━━━━━━━━━━━\n" +
      "Signal: %s\n" +
      "Buy: %.2f%%\n" +
      "Sell: %.2f%%\n" +
      "Confidence: %.2f%%\n" +
      "Regime: %s\n" +
      "━━━━━━━━━━━━━━━━━\n" +
      "Time: %s",
      signal,
      buy_prob * 100,
      sell_prob * 100,
      confidence * 100,
      regime,
      TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES)
   );
   
   //--- Create or update label
   if(ObjectFind(0, label_name) < 0)
   {
      ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, label_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 10);
      ObjectSetString(0, label_name, OBJPROP_FONT, "Courier New");
   }
   
   ObjectSetString(0, label_name, OBJPROP_TEXT, info);
   ObjectSetInteger(0, label_name, OBJPROP_COLOR, signal_color);
   
   ChartRedraw();
}

//+------------------------------------------------------------------+
