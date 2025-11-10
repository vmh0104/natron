//+------------------------------------------------------------------+
//|                                              natron_ea.mq5       |
//|                        Natron Transformer Trading EA              |
//|                    Connects to Python Socket Server              |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property link      ""
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "127.0.0.1";      // Python server host
input int      ServerPort = 8888;              // Python server port
input double   LotSize = 0.01;                 // Lot size
input int      MagicNumber = 123456;           // Magic number
input int      MaxSlippage = 10;               // Max slippage (points)
input bool     UseStopLoss = true;             // Use stop loss
input double   StopLossPips = 50;              // Stop loss (pips)
input bool     UseTakeProfit = true;           // Use take profit
input double   TakeProfitPips = 100;           // Take profit (pips)
input int      SequenceLength = 96;            // Number of candles for prediction
input int      UpdateIntervalSeconds = 60;     // Update interval (seconds)
input double   MinBuyProbability = 0.6;        // Minimum buy probability
input double   MinSellProbability = 0.6;       // Minimum sell probability
input double   MinConfidence = 0.5;            // Minimum confidence threshold

//--- Global variables
CTrade trade;
int socketHandle = INVALID_HANDLE;
datetime lastUpdateTime = 0;
double lastBuyProb = 0.0;
double lastSellProb = 0.0;
string lastRegime = "";
double lastConfidence = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set trade parameters
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(MaxSlippage);
   trade.SetTypeFilling(ORDER_FILLING_FOK);
   trade.SetAsyncMode(false);
   
   // Connect to Python server
   if(!ConnectToServer())
   {
      Print("Failed to connect to Python server. EA will retry on next update.");
   }
   
   Print("Natron EA initialized");
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
   datetime currentTime = TimeCurrent();
   if(currentTime - lastUpdateTime < UpdateIntervalSeconds)
      return;
   
   lastUpdateTime = currentTime;
   
   // Ensure connection
   if(socketHandle == INVALID_HANDLE)
   {
      if(!ConnectToServer())
      {
         Print("Failed to connect to server. Skipping this update.");
         return;
      }
   }
   
   // Get prediction from AI model
   if(!GetPrediction())
   {
      Print("Failed to get prediction. Disconnecting and will retry.");
      DisconnectFromServer();
      return;
   }
   
   // Execute trading logic
   ExecuteTradingLogic();
}

//+------------------------------------------------------------------+
//| Connect to Python socket server                                  |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
   socketHandle = SocketCreate();
   if(socketHandle == INVALID_HANDLE)
   {
      Print("Failed to create socket");
      return false;
   }
   
   if(!SocketConnect(socketHandle, ServerHost, ServerPort, 1000))
   {
      Print("Failed to connect to ", ServerHost, ":", ServerPort);
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      return false;
   }
   
   Print("Connected to Python server at ", ServerHost, ":", ServerPort);
   return true;
}

//+------------------------------------------------------------------+
//| Disconnect from server                                            |
//+------------------------------------------------------------------+
void DisconnectFromServer()
{
   if(socketHandle != INVALID_HANDLE)
   {
      SocketClose(socketHandle);
      socketHandle = INVALID_HANDLE;
      Print("Disconnected from server");
   }
}

//+------------------------------------------------------------------+
//| Get prediction from AI model                                     |
//+------------------------------------------------------------------+
bool GetPrediction()
{
   // Collect OHLCV data
   MqlRates rates[];
   int copied = CopyRates(_Symbol, PERIOD_CURRENT, 0, SequenceLength, rates);
   if(copied < SequenceLength)
   {
      Print("Failed to copy rates. Got ", copied, " candles, need ", SequenceLength);
      return false;
   }
   
   // Build JSON request
   string jsonRequest = BuildPredictionRequest(rates);
   
   // Send request
   uchar request[];
   StringToCharArray(jsonRequest, request);
   if(SocketSend(socketHandle, request) <= 0)
   {
      Print("Failed to send request");
      return false;
   }
   
   // Receive response
   uchar response[];
   int timeout = 5000; // 5 seconds
   int received = SocketRead(socketHandle, response, timeout);
   if(received <= 0)
   {
      Print("Failed to receive response or timeout");
      return false;
   }
   
   // Parse response
   string jsonResponse = CharArrayToString(response);
   if(!ParsePredictionResponse(jsonResponse))
   {
      Print("Failed to parse response: ", jsonResponse);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Build prediction request JSON                                    |
//+------------------------------------------------------------------+
string BuildPredictionRequest(MqlRates &rates[])
{
   string json = "{\"type\":\"predict\",\"candles\":[";
   
   for(int i = 0; i < ArraySize(rates); i++)
   {
      if(i > 0) json += ",";
      json += "{";
      json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
      json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
      json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
      json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
      json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
      json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
      json += "}";
   }
   
   json += "]}\n";
   return json;
}

//+------------------------------------------------------------------+
//| Parse prediction response JSON                                   |
//+------------------------------------------------------------------+
bool ParsePredictionResponse(string jsonResponse)
{
   // Simple JSON parsing (for production, consider using a JSON library)
   // Extract buy_prob, sell_prob, regime, confidence
   
   int buyPos = StringFind(jsonResponse, "\"buy_prob\":");
   int sellPos = StringFind(jsonResponse, "\"sell_prob\":");
   int regimePos = StringFind(jsonResponse, "\"regime\":");
   int confPos = StringFind(jsonResponse, "\"confidence\":");
   
   if(buyPos < 0 || sellPos < 0 || regimePos < 0 || confPos < 0)
   {
      Print("Missing fields in response");
      return false;
   }
   
   // Extract values (simplified parsing)
   string buyStr = ExtractValue(jsonResponse, buyPos);
   string sellStr = ExtractValue(jsonResponse, sellPos);
   string regimeStr = ExtractStringValue(jsonResponse, regimePos);
   string confStr = ExtractValue(jsonResponse, confPos);
   
   lastBuyProb = StringToDouble(buyStr);
   lastSellProb = StringToDouble(sellStr);
   lastRegime = regimeStr;
   lastConfidence = StringToDouble(confStr);
   
   Print("Prediction - Buy: ", lastBuyProb, ", Sell: ", lastSellProb, 
         ", Regime: ", lastRegime, ", Confidence: ", lastConfidence);
   
   return true;
}

//+------------------------------------------------------------------+
//| Extract numeric value from JSON                                  |
//+------------------------------------------------------------------+
string ExtractValue(string json, int startPos)
{
   int colonPos = StringFind(json, ":", startPos);
   if(colonPos < 0) return "0";
   
   int valueStart = colonPos + 1;
   int valueEnd = valueStart;
   
   // Find end of number
   while(valueEnd < StringLen(json) && 
         (StringGetCharacter(json, valueEnd) >= '0' && StringGetCharacter(json, valueEnd) <= '9' ||
          StringGetCharacter(json, valueEnd) == '.' || StringGetCharacter(json, valueEnd) == '-'))
   {
      valueEnd++;
   }
   
   // Skip comma or closing brace
   while(valueEnd < StringLen(json) && 
         (StringGetCharacter(json, valueEnd) == ',' || StringGetCharacter(json, valueEnd) == '}'))
   {
      valueEnd++;
   }
   
   return StringSubstr(json, valueStart, valueEnd - valueStart);
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                   |
//+------------------------------------------------------------------+
string ExtractStringValue(string json, int startPos)
{
   int colonPos = StringFind(json, ":", startPos);
   if(colonPos < 0) return "";
   
   int quoteStart = StringFind(json, "\"", colonPos);
   if(quoteStart < 0) return "";
   
   int quoteEnd = StringFind(json, "\"", quoteStart + 1);
   if(quoteEnd < 0) return "";
   
   return StringSubstr(json, quoteStart + 1, quoteEnd - quoteStart - 1);
}

//+------------------------------------------------------------------+
//| Execute trading logic based on predictions                       |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
   // Check confidence threshold
   if(lastConfidence < MinConfidence)
   {
      Print("Confidence too low: ", lastConfidence, " < ", MinConfidence);
      return;
   }
   
   // Check current position
   bool hasPosition = PositionSelect(_Symbol);
   int positionType = -1; // -1: no position, 0: buy, 1: sell
   
   if(hasPosition)
   {
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         positionType = 0;
      else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
         positionType = 1;
   }
   
   // Trading logic
   if(lastBuyProb >= MinBuyProbability && positionType != 0)
   {
      // Close sell position if exists
      if(positionType == 1)
      {
         trade.PositionClose(_Symbol);
         Print("Closed sell position before opening buy");
      }
      
      // Open buy position
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = UseStopLoss ? price - StopLossPips * _Point * 10 : 0;
      double tp = UseTakeProfit ? price + TakeProfitPips * _Point * 10 : 0;
      
      if(trade.Buy(LotSize, _Symbol, price, sl, tp, "Natron Buy Signal"))
      {
         Print("Opened BUY position. Buy prob: ", lastBuyProb, ", Regime: ", lastRegime);
      }
      else
      {
         Print("Failed to open BUY position. Error: ", GetLastError());
      }
   }
   else if(lastSellProb >= MinSellProbability && positionType != 1)
   {
      // Close buy position if exists
      if(positionType == 0)
      {
         trade.PositionClose(_Symbol);
         Print("Closed buy position before opening sell");
      }
      
      // Open sell position
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = UseStopLoss ? price + StopLossPips * _Point * 10 : 0;
      double tp = UseTakeProfit ? price - TakeProfitPips * _Point * 10 : 0;
      
      if(trade.Sell(LotSize, _Symbol, price, sl, tp, "Natron Sell Signal"))
      {
         Print("Opened SELL position. Sell prob: ", lastSellProb, ", Regime: ", lastRegime);
      }
      else
      {
         Print("Failed to open SELL position. Error: ", GetLastError());
      }
   }
   else
   {
      // No strong signal - could close positions if desired
      // For now, we keep existing positions
   }
}

//+------------------------------------------------------------------+
