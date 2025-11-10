//+------------------------------------------------------------------+
//|                                              Natron_EA_v2.mq5   |
//|                        Natron Transformer Trading EA             |
//+------------------------------------------------------------------+
#property copyright "Natron AI"
#property version   "2.00"
#property description "Multi-task Transformer model for financial trading"

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "localhost";      // Python server host
input int      ServerPort = 8888;             // Python server port
input double   LotSize = 0.01;                // Lot size
input int      MagicNumber = 123456;          // Magic number
input int      StopLoss = 100;                // Stop loss (points)
input int      TakeProfit = 200;              // Take profit (points)
input double   MinConfidence = 0.6;           // Minimum confidence threshold
input int      SequenceLength = 96;           // Sequence length for model
input int      UpdateInterval = 60;           // Update interval (seconds)
input bool     UseBuySignals = true;          // Use buy signals
input bool     UseSellSignals = true;          // Use sell signals

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
   trade.SetAsyncMode(false);
   
   Print("Natron EA initialized");
   Print("Connecting to Python server at ", ServerHost, ":", ServerPort);
   
   // Connect to socket server
   if(!ConnectToServer())
   {
      Print("Failed to connect to server. Retrying...");
      return(INIT_FAILED);
   }
   
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
   datetime current_time = TimeCurrent();
   if(current_time - last_update < UpdateInterval)
      return;
   
   last_update = current_time;
   
   // Get prediction from server
   if(!GetPrediction())
   {
      Print("Failed to get prediction from server");
      return;
   }
   
   // Check if we should trade
   if(last_confidence < MinConfidence)
   {
      Print("Confidence too low: ", last_confidence);
      return;
   }
   
   // Trading logic
   ManagePositions();
   
   // Open new positions if signals are strong
   if(UseBuySignals && last_buy_prob > 0.7 && last_sell_prob < 0.3)
   {
      OpenBuyPosition();
   }
   
   if(UseSellSignals && last_sell_prob > 0.7 && last_buy_prob < 0.3)
   {
      OpenSellPosition();
   }
}

//+------------------------------------------------------------------+
//| Connect to Python socket server                                  |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socket_handle = SocketCreate();
   if(socket_handle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return false;
   }
   
   if(!SocketConnect(socket_handle, ServerHost, ServerPort, 5000))
   {
      Print("Failed to connect to ", ServerHost, ":", ServerPort);
      SocketClose(socket_handle);
      socket_handle = INVALID_HANDLE;
      return false;
   }
   
   Print("Connected to server successfully");
   return true;
}

//+------------------------------------------------------------------+
//| Disconnect from server                                            |
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
//| Get prediction from server                                       |
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
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, SequenceLength, rates);
   if(copied < SequenceLength)
   {
      Print("Not enough historical data. Need ", SequenceLength, " candles");
      return false;
   }
   
   // Build JSON request
   string json_request = BuildJSONRequest(rates);
   
   // Send request
   char request[];
   StringToCharArray(json_request, request, 0, StringLen(json_request));
   
   if(!SocketSend(socket_handle, request))
   {
      Print("Failed to send request");
      DisconnectFromServer();
      return false;
   }
   
   // Receive response
   char response[];
   int timeout = 5000; // 5 seconds
   int received = SocketRead(socket_handle, response, timeout);
   
   if(received <= 0)
   {
      Print("Failed to receive response or timeout");
      DisconnectFromServer();
      return false;
   }
   
   // Parse response
   string response_str = CharArrayToString(response);
   if(!ParseJSONResponse(response_str))
   {
      Print("Failed to parse response: ", response_str);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Build JSON request from rates                                    |
//+------------------------------------------------------------------+
string BuildJSONRequest(MqlRates &rates[])
{
   string json = "{\"action\":\"predict\",\"candles\":[";
   
   for(int i = 0; i < ArraySize(rates); i++)
   {
      if(i > 0) json += ",";
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
//| Parse JSON response                                               |
//+------------------------------------------------------------------+
bool ParseJSONResponse(string response)
{
   // Simple JSON parsing (for production, use proper JSON library)
   // Expected format: {"status":"success","result":{"buy_prob":0.71,...}}
   
   int status_pos = StringFind(response, "\"status\"");
   if(status_pos < 0) return false;
   
   int status_start = StringFind(response, ":", status_pos) + 1;
   int status_end = StringFind(response, ",", status_start);
   if(status_end < 0) status_end = StringFind(response, "}", status_start);
   
   string status = StringSubstr(response, status_start, status_end - status_start);
   StringReplace(status, "\"", "");
   StringTrimLeft(status);
   StringTrimRight(status);
   
   if(status != "success") return false;
   
   // Extract buy_prob
   int buy_pos = StringFind(response, "\"buy_prob\"");
   if(buy_pos >= 0)
   {
      int buy_start = StringFind(response, ":", buy_pos) + 1;
      int buy_end = StringFind(response, ",", buy_start);
      if(buy_end < 0) buy_end = StringFind(response, "}", buy_start);
      string buy_str = StringSubstr(response, buy_start, buy_end - buy_start);
      last_buy_prob = StringToDouble(buy_str);
   }
   
   // Extract sell_prob
   int sell_pos = StringFind(response, "\"sell_prob\"");
   if(sell_pos >= 0)
   {
      int sell_start = StringFind(response, ":", sell_pos) + 1;
      int sell_end = StringFind(response, ",", sell_start);
      if(sell_end < 0) sell_end = StringFind(response, "}", sell_start);
      string sell_str = StringSubstr(response, sell_start, sell_end - sell_start);
      last_sell_prob = StringToDouble(sell_str);
   }
   
   // Extract confidence
   int conf_pos = StringFind(response, "\"confidence\"");
   if(conf_pos >= 0)
   {
      int conf_start = StringFind(response, ":", conf_pos) + 1;
      int conf_end = StringFind(response, ",", conf_start);
      if(conf_end < 0) conf_end = StringFind(response, "}", conf_start);
      string conf_str = StringSubstr(response, conf_start, conf_end - conf_start);
      last_confidence = StringToDouble(conf_str);
   }
   
   // Extract regime
   int regime_pos = StringFind(response, "\"regime\"");
   if(regime_pos >= 0)
   {
      int regime_start = StringFind(response, "\"", regime_pos + 8) + 1;
      int regime_end = StringFind(response, "\"", regime_start);
      last_regime = StringSubstr(response, regime_start, regime_end - regime_start);
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void ManagePositions()
{
   // Close positions if signals reverse
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      
      // Close buy if sell signal is strong
      if(pos_type == POSITION_TYPE_BUY && last_sell_prob > 0.7)
      {
         trade.PositionClose(ticket);
         Print("Closed buy position due to sell signal");
      }
      
      // Close sell if buy signal is strong
      if(pos_type == POSITION_TYPE_SELL && last_buy_prob > 0.7)
      {
         trade.PositionClose(ticket);
         Print("Closed sell position due to buy signal");
      }
   }
}

//+------------------------------------------------------------------+
//| Open buy position                                                 |
//+------------------------------------------------------------------+
void OpenBuyPosition()
{
   // Check if we already have a buy position
   if(PositionSelect(_Symbol))
   {
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         return; // Already have buy position
   }
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl = StopLoss > 0 ? ask - StopLoss * _Point : 0;
   double tp = TakeProfit > 0 ? ask + TakeProfit * _Point : 0;
   
   if(trade.Buy(LotSize, _Symbol, ask, sl, tp, "Natron Buy"))
   {
      Print("Buy order opened. Buy prob: ", last_buy_prob, ", Regime: ", last_regime);
   }
   else
   {
      Print("Failed to open buy order: ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Open sell position                                                |
//+------------------------------------------------------------------+
void OpenSellPosition()
{
   // Check if we already have a sell position
   if(PositionSelect(_Symbol))
   {
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
         return; // Already have sell position
   }
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = StopLoss > 0 ? bid + StopLoss * _Point : 0;
   double tp = TakeProfit > 0 ? bid - TakeProfit * _Point : 0;
   
   if(trade.Sell(LotSize, _Symbol, bid, sl, tp, "Natron Sell"))
   {
      Print("Sell order opened. Sell prob: ", last_sell_prob, ", Regime: ", last_regime);
   }
   else
   {
      Print("Failed to open sell order: ", trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
