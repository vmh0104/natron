//+------------------------------------------------------------------+
//|                                                  natron_ea.mq5   |
//|                        Natron Transformer MQL5 Expert Advisor    |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property version   "2.0"
#property description "Expert Advisor for Natron Transformer AI Model"
#property description "Connects to Python inference server via TCP socket"

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "localhost";      // Python server host
input int      ServerPort = 8888;            // Python server port
input double   LotSize = 0.01;                // Lot size
input int      MagicNumber = 123456;          // Magic number
input int      StopLoss = 50;                 // Stop loss (points)
input int      TakeProfit = 100;              // Take profit (points)
input double   BuyThreshold = 0.65;           // Buy signal threshold
input double   SellThreshold = 0.65;          // Sell signal threshold
input int      CandleCount = 96;               // Number of candles for prediction
input int      UpdateInterval = 60;            // Update interval (seconds)
input bool     EnableTrading = true;          // Enable trading
input bool     ShowSignals = true;             // Show signals on chart

//--- Global variables
CTrade trade;
int socket_handle = INVALID_HANDLE;
datetime last_update = 0;
double last_buy_prob = 0.0;
double last_sell_prob = 0.0;
string last_regime = "";
double last_confidence = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(10);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   
   // Connect to Python server
   if(!ConnectToServer())
   {
      Print("Failed to connect to Python server. Please ensure server_natron.py is running.");
      return(INIT_FAILED);
   }
   
   Print("Natron EA initialized successfully");
   Print("Connected to Python server at ", ServerHost, ":", ServerPort);
   
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
   // Check if it's time to update
   if(TimeCurrent() - last_update < UpdateInterval)
      return;
   
   last_update = TimeCurrent();
   
   // Get prediction from server
   if(!GetPrediction())
   {
      Print("Failed to get prediction from server");
      return;
   }
   
   // Display signals on chart
   if(ShowSignals)
   {
      DisplaySignals();
   }
   
   // Execute trading logic
   if(EnableTrading)
   {
      ExecuteTradingLogic();
   }
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
   
   if(!SocketConnect(socket_handle, ServerHost, ServerPort, 1000))
   {
      Print("Failed to connect to ", ServerHost, ":", ServerPort);
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Disconnect from server                                           |
//+------------------------------------------------------------------+
void DisconnectFromServer()
{
   if(socket_handle != INVALID_HANDLE)
   {
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
//| Get prediction from Python server                                |
//+------------------------------------------------------------------+
bool GetPrediction()
{
   if(socket_handle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
         return false;
   }
   
   // Prepare candles data
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, CandleCount, rates) < CandleCount)
   {
      Print("Failed to get candle data");
      return false;
   }
   
   // Build JSON request
   string json_request = BuildJSONRequest(rates);
   
   // Send request
   uchar request[];
   StringToCharArray(json_request, request, 0, StringLen(json_request));
   
   if(SocketSend(socket_handle, request) <= 0)
   {
      Print("Failed to send request");
      DisconnectFromServer();
      return false;
   }
   
   // Receive response
   uchar response[];
   int timeout = 1000;
   int received = SocketRead(socket_handle, response, timeout);
   
   if(received <= 0)
   {
      Print("Failed to receive response");
      DisconnectFromServer();
      return false;
   }
   
   // Parse response
   string json_response = CharArrayToString(response);
   if(!ParseJSONResponse(json_response))
   {
      Print("Failed to parse response: ", json_response);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Build JSON request                                               |
//+------------------------------------------------------------------+
string BuildJSONRequest(MqlRates &rates[])
{
   string json = "{\"action\":\"predict\",\"candles\":[";
   
   for(int i = ArraySize(rates) - 1; i >= 0; i--)
   {
      if(i < ArraySize(rates) - 1)
         json += ",";
      
      json += "{";
      json += "\"time\":" + IntegerToString(rates[i].time) + ",";
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
//| Parse JSON response                                              |
//+------------------------------------------------------------------+
bool ParseJSONResponse(string json_response)
{
   // Simple JSON parsing (for production, use proper JSON library)
   int buy_pos = StringFind(json_response, "\"buy_prob\":");
   int sell_pos = StringFind(json_response, "\"sell_prob\":");
   int regime_pos = StringFind(json_response, "\"regime\":");
   int conf_pos = StringFind(json_response, "\"confidence\":");
   
   if(buy_pos < 0 || sell_pos < 0 || regime_pos < 0 || conf_pos < 0)
      return false;
   
   // Extract buy_prob
   string buy_str = StringSubstr(json_response, buy_pos + 11);
   int comma_pos = StringFind(buy_str, ",");
   if(comma_pos > 0)
      buy_str = StringSubstr(buy_str, 0, comma_pos);
   last_buy_prob = StringToDouble(buy_str);
   
   // Extract sell_prob
   string sell_str = StringSubstr(json_response, sell_pos + 12);
   comma_pos = StringFind(sell_str, ",");
   if(comma_pos > 0)
      sell_str = StringSubstr(sell_str, 0, comma_pos);
   last_sell_prob = StringToDouble(sell_str);
   
   // Extract regime
   string regime_str = StringSubstr(json_response, regime_pos + 10);
   comma_pos = StringFind(regime_str, ",");
   if(comma_pos > 0)
      regime_str = StringSubstr(regime_str, 0, comma_pos);
   regime_str = StringSubstr(regime_str, 1, StringLen(regime_str) - 2); // Remove quotes
   last_regime = regime_str;
   
   // Extract confidence
   string conf_str = StringSubstr(json_response, conf_pos + 13);
   comma_pos = StringFind(conf_str, ",");
   if(comma_pos > 0)
      conf_str = StringSubstr(conf_str, 0, comma_pos);
   last_confidence = StringToDouble(conf_str);
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic                                            |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
   // Check existing positions
   bool has_buy = PositionSelect(_Symbol);
   bool has_sell = false;
   
   if(has_buy)
   {
      long pos_type = PositionGetInteger(POSITION_TYPE);
      has_buy = (pos_type == POSITION_TYPE_BUY);
      has_sell = (pos_type == POSITION_TYPE_SELL);
   }
   
   // Buy signal
   if(last_buy_prob >= BuyThreshold && !has_buy && last_confidence > 0.6)
   {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = StopLoss > 0 ? price - StopLoss * _Point : 0;
      double tp = TakeProfit > 0 ? price + TakeProfit * _Point : 0;
      
      if(trade.Buy(LotSize, _Symbol, price, sl, tp, "Natron Buy Signal"))
      {
         Print("Buy order opened. Buy Prob: ", last_buy_prob, ", Regime: ", last_regime);
      }
   }
   
   // Sell signal
   if(last_sell_prob >= SellThreshold && !has_sell && last_confidence > 0.6)
   {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = StopLoss > 0 ? price + StopLoss * _Point : 0;
      double tp = TakeProfit > 0 ? price - TakeProfit * _Point : 0;
      
      if(trade.Sell(LotSize, _Symbol, price, sl, tp, "Natron Sell Signal"))
      {
         Print("Sell order opened. Sell Prob: ", last_sell_prob, ", Regime: ", last_regime);
      }
   }
   
   // Close positions if signal reverses
   if(has_buy && last_sell_prob > last_buy_prob && last_sell_prob > 0.7)
   {
      if(trade.PositionClose(_Symbol))
         Print("Buy position closed due to sell signal");
   }
   
   if(has_sell && last_buy_prob > last_sell_prob && last_buy_prob > 0.7)
   {
      if(trade.PositionClose(_Symbol))
         Print("Sell position closed due to buy signal");
   }
}

//+------------------------------------------------------------------+
//| Display signals on chart                                         |
//+------------------------------------------------------------------+
void DisplaySignals()
{
   string info = "\n=== NATRON AI SIGNALS ===\n";
   info += "Buy Probability: " + DoubleToString(last_buy_prob * 100, 2) + "%\n";
   info += "Sell Probability: " + DoubleToString(last_sell_prob * 100, 2) + "%\n";
   info += "Market Regime: " + last_regime + "\n";
   info += "Confidence: " + DoubleToString(last_confidence * 100, 2) + "%\n";
   info += "========================";
   
   Comment(info);
}

//+------------------------------------------------------------------+
