//+------------------------------------------------------------------+
//|                                                  natron_ea.mq5   |
//|                        Natron Transformer Trading EA for MT5    |
//|                        Connects to Python Socket Server         |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property link      ""
#property version   "2.00"
#property description "Natron Transformer - Multi-Task Financial Trading Model"
#property description "Connects to Python GPU server via TCP socket"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

//--- Input parameters
input string   ServerHost = "localhost";        // Python server host
input int      ServerPort = 8888;               // Python server port
input double   LotSize = 0.01;                  // Lot size
input int      MagicNumber = 123456;           // Magic number
input int      SequenceLength = 96;             // Number of candles for prediction
input int      PredictionInterval = 15;         // Minutes between predictions
input double   BuyThreshold = 0.6;              // Buy signal threshold
input double   SellThreshold = 0.6;             // Sell signal threshold
input bool     UseStopLoss = true;              // Use stop loss
input double   StopLossPips = 50;               // Stop loss in pips
input bool     UseTakeProfit = true;            // Use take profit
input double   TakeProfitPips = 100;            // Take profit in pips
input int      SocketTimeout = 5000;           // Socket timeout (ms)
input bool     EnableLogging = true;            // Enable logging

//--- Global variables
CTrade trade;
CPositionInfo position;
CAccountInfo account;

int socket_handle = INVALID_HANDLE;
datetime last_prediction_time = 0;
MqlRates rates[];
double last_buy_prob = 0.0;
double last_sell_prob = 0.0;
string last_regime = "";

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
   }
   
   // Initialize arrays
   ArraySetAsSeries(rates, true);
   
   Print("Natron EA initialized. Waiting for predictions...");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   DisconnectFromServer();
   Print("Natron EA deinitialized.");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if it's time for a new prediction
   datetime current_time = TimeCurrent();
   if(current_time - last_prediction_time < PredictionInterval * 60)
   {
      return; // Too soon for next prediction
   }
   
   // Ensure connection
   if(socket_handle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
      {
         return; // Connection failed, try again next tick
      }
   }
   
   // Get recent candles
   if(!GetCandles(SequenceLength))
   {
      Print("Failed to get candles");
      return;
   }
   
   // Send prediction request
   string prediction = RequestPrediction();
   if(prediction == "")
   {
      Print("Failed to get prediction");
      return;
   }
   
   // Parse prediction response
   if(!ParsePrediction(prediction))
   {
      Print("Failed to parse prediction");
      return;
   }
   
   // Update last prediction time
   last_prediction_time = current_time;
   
   // Execute trading logic
   ExecuteTradingLogic();
}

//+------------------------------------------------------------------+
//| Connect to Python socket server                                 |
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
      Print("Failed to connect to ", ServerHost, ":", ServerPort);
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   Print("Connected to Python server at ", ServerHost, ":", ServerPort);
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
//| Get recent candles                                               |
//+------------------------------------------------------------------+
bool GetCandles(int count)
{
   ArrayResize(rates, count);
   
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, count, rates) != count)
   {
      Print("Failed to copy rates");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Request prediction from Python server                            |
//+------------------------------------------------------------------+
string RequestPrediction()
{
   // Build JSON request
   string json_request = BuildPredictionRequest();
   
   // Send request
   uchar request_data[];
   StringToCharArray(json_request, request_data, 0, StringLen(json_request));
   
   if(SocketSend(socket_handle, request_data) <= 0)
   {
      Print("Failed to send request");
      DisconnectFromServer();
      return "";
   }
   
   // Receive response
   uchar response_data[];
   int timeout = 5000; // 5 seconds
   int received = SocketRead(socket_handle, response_data, timeout);
   
   if(received <= 0)
   {
      Print("Failed to receive response or timeout");
      DisconnectFromServer();
      return "";
   }
   
   string response = CharArrayToString(response_data, 0, received);
   return response;
}

//+------------------------------------------------------------------+
//| Build prediction request JSON                                    |
//+------------------------------------------------------------------+
string BuildPredictionRequest()
{
   string json = "{\"type\":\"predict\",\"candles\":[";
   
   for(int i = 0; i < ArraySize(rates); i++)
   {
      if(i > 0) json += ",";
      
      json += "{";
      json += "\"time\":" + IntegerToString((int)rates[i].time) + ",";
      json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString((int)rates[i].tick_volume);
      json += "}";
   }
   
   json += "]}";
   return json;
}

//+------------------------------------------------------------------+
//| Parse prediction response                                        |
//+------------------------------------------------------------------+
bool ParsePrediction(string response)
{
   // Simple JSON parsing (for production, use a proper JSON library)
   // Extract buy_prob, sell_prob, direction_up, regime
   
   int buy_pos = StringFind(response, "\"buy_prob\":");
   int sell_pos = StringFind(response, "\"sell_prob\":");
   int regime_pos = StringFind(response, "\"regime\":");
   
   if(buy_pos < 0 || sell_pos < 0 || regime_pos < 0)
   {
      Print("Invalid prediction response format");
      return false;
   }
   
   // Extract buy probability
   string buy_str = StringSubstr(response, buy_pos + 11, 10);
   last_buy_prob = StringToDouble(buy_str);
   
   // Extract sell probability
   string sell_str = StringSubstr(response, sell_pos + 12, 10);
   last_sell_prob = StringToDouble(sell_str);
   
   // Extract regime
   int regime_start = StringFind(response, "\"regime\":\"", regime_pos) + 10;
   int regime_end = StringFind(response, "\"", regime_start);
   if(regime_end > regime_start)
   {
      last_regime = StringSubstr(response, regime_start, regime_end - regime_start);
   }
   
   if(EnableLogging)
   {
      Print("Prediction: Buy=", DoubleToString(last_buy_prob, 4), 
            ", Sell=", DoubleToString(last_sell_prob, 4),
            ", Regime=", last_regime);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic based on predictions                       |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
   // Check current position
   bool has_position = position.Select(_Symbol);
   ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)-1;
   
   if(has_position)
   {
      pos_type = (ENUM_POSITION_TYPE)position.PositionType();
   }
   
   // Calculate stop loss and take profit
   double sl = 0, tp = 0;
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if(UseStopLoss)
   {
      sl = price - StopLossPips * point * 10; // Adjust for 5-digit brokers
   }
   if(UseTakeProfit)
   {
      tp = price + TakeProfitPips * point * 10;
   }
   
   // Buy signal
   if(last_buy_prob >= BuyThreshold && pos_type != POSITION_TYPE_BUY)
   {
      // Close sell position if exists
      if(pos_type == POSITION_TYPE_SELL)
      {
         trade.PositionClose(_Symbol);
      }
      
      // Open buy position
      if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "Natron Buy Signal"))
      {
         Print("Buy order opened. Buy prob: ", DoubleToString(last_buy_prob, 4));
      }
      else
      {
         Print("Failed to open buy order. Error: ", trade.ResultRetcodeDescription());
      }
   }
   
   // Sell signal
   if(last_sell_prob >= SellThreshold && pos_type != POSITION_TYPE_SELL)
   {
      // Close buy position if exists
      if(pos_type == POSITION_TYPE_BUY)
      {
         trade.PositionClose(_Symbol);
      }
      
      // Adjust SL/TP for sell
      if(UseStopLoss)
      {
         sl = price + StopLossPips * point * 10;
      }
      if(UseTakeProfit)
      {
         tp = price - TakeProfitPips * point * 10;
      }
      
      // Open sell position
      if(trade.Sell(LotSize, _Symbol, 0, sl, tp, "Natron Sell Signal"))
      {
         Print("Sell order opened. Sell prob: ", DoubleToString(last_sell_prob, 4));
      }
      else
      {
         Print("Failed to open sell order. Error: ", trade.ResultRetcodeDescription());
      }
   }
   
   // Exit conditions (optional: exit on opposite signal)
   if(has_position)
   {
      if(pos_type == POSITION_TYPE_BUY && last_sell_prob > last_buy_prob + 0.2)
      {
         trade.PositionClose(_Symbol);
         Print("Closed buy position due to sell signal");
      }
      else if(pos_type == POSITION_TYPE_SELL && last_buy_prob > last_sell_prob + 0.2)
      {
         trade.PositionClose(_Symbol);
         Print("Closed sell position due to buy signal");
      }
   }
}

//+------------------------------------------------------------------+
//| Timer function (optional)                                       |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Can be used for periodic health checks
   if(socket_handle == INVALID_HANDLE)
   {
      ConnectToServer();
   }
}
