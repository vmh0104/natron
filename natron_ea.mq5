//+------------------------------------------------------------------+
//|                                              natron_ea.mq5       |
//|                        Natron Transformer MQL5 Expert Advisor    |
//|                        Connects to Python Socket Server          |
//+------------------------------------------------------------------+
#property copyright "Natron AI Trading System"
#property version   "2.0"
#property description "Natron Transformer - Multi-Task Financial Trading Model"
#property description "Connects to Python GPU server for AI predictions"

#include <Trade\Trade.mqh>

//--- Input parameters
input string   ServerHost = "localhost";        // Python Server Host
input int      ServerPort = 8888;              // Python Server Port
input double   LotSize = 0.01;                 // Lot Size
input int      MagicNumber = 123456;           // Magic Number
input int      StopLoss = 50;                  // Stop Loss (points)
input int      TakeProfit = 100;               // Take Profit (points)
input double   BuyThreshold = 0.6;             // Buy Signal Threshold
input double   SellThreshold = 0.6;           // Sell Signal Threshold
input int      SequenceLength = 96;            // Required Candles for Prediction
input int      UpdateInterval = 60;            // Update Interval (seconds)
input bool     EnableTrading = true;           // Enable Trading
input bool     UseRegimeFilter = true;         // Use Regime Filter
input string   AllowedRegimes = "BULL_STRONG,BULL_WEAK"; // Allowed Regimes

//--- Global variables
int socketHandle = INVALID_HANDLE;
datetime lastUpdateTime = 0;
string lastPrediction = "";
double lastBuyProb = 0.0;
double lastSellProb = 0.0;
string lastRegime = "";
CTrade trade;

//--- Regime names
string REGIME_NAMES[] = {
    "BULL_STRONG",
    "BULL_WEAK",
    "RANGE",
    "BEAR_WEAK",
    "BEAR_STRONG",
    "VOLATILE"
};

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
        Print("Failed to connect to Python server. EA will retry...");
        return(INIT_FAILED);
    }
    
    Print("Natron EA initialized successfully");
    Print("Server: ", ServerHost, ":", ServerPort);
    Print("Symbol: ", _Symbol, ", Timeframe: ", EnumToString((ENUM_TIMEFRAMES)_Period));
    
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
    // Check if enough time has passed
    datetime currentTime = TimeCurrent();
    if(currentTime - lastUpdateTime < UpdateInterval)
        return;
    
    // Check if we have enough candles
    if(Bars(_Symbol, _Period) < SequenceLength)
    {
        Print("Not enough bars. Need: ", SequenceLength, ", Have: ", Bars(_Symbol, _Period));
        return;
    }
    
    // Get prediction from server
    if(!GetPrediction())
    {
        Print("Failed to get prediction");
        return;
    }
    
    lastUpdateTime = currentTime;
    
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
    
    Print("Connected to Python server");
    
    // Send status request
    SendStatusRequest();
    
    return true;
}

//+------------------------------------------------------------------+
//| Disconnect from server                                           |
//+------------------------------------------------------------------+
void DisconnectFromServer()
{
    if(socketHandle != INVALID_HANDLE)
    {
        SocketClose(socketHandle);
        socketHandle = INVALID_HANDLE;
    }
}

//+------------------------------------------------------------------+
//| Send status request                                              |
//+------------------------------------------------------------------+
void SendStatusRequest()
{
    string message = "{\"type\":\"status\"}\n";
    SocketSend(socketHandle, message);
}

//+------------------------------------------------------------------+
//| Get prediction from server                                       |
//+------------------------------------------------------------------+
bool GetPrediction()
{
    // Prepare candle data
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    
    if(CopyRates(_Symbol, _Period, 0, SequenceLength, rates) < SequenceLength)
    {
        Print("Failed to copy rates");
        return false;
    }
    
    // Build JSON message
    string json = BuildCandleJSON(rates);
    
    // Send prediction request
    string request = "{\"type\":\"predict\",\"candles\":" + json + "}\n";
    
    if(!SocketSend(socketHandle, request))
    {
        Print("Failed to send prediction request");
        // Try to reconnect
        DisconnectFromServer();
        if(!ConnectToServer())
            return false;
        return false;
    }
    
    // Receive response
    string response = "";
    char data[];
    uint timeout = 5000; // 5 seconds
    
    ulong startTime = GetTickCount64();
    while(GetTickCount64() - startTime < timeout)
    {
        uint received = SocketRead(socketHandle, data, 0, 1000);
        if(received > 0)
        {
            response += CharArrayToString(data, 0, received);
            if(StringFind(response, "\n") >= 0)
                break;
        }
        Sleep(10);
    }
    
    if(StringLen(response) == 0)
    {
        Print("No response from server");
        return false;
    }
    
    // Parse response
    return ParsePredictionResponse(response);
}

//+------------------------------------------------------------------+
//| Build candle JSON                                                |
//+------------------------------------------------------------------+
string BuildCandleJSON(MqlRates &rates[])
{
    string json = "[";
    
    for(int i = ArraySize(rates) - 1; i >= 0; i--)
    {
        if(i < ArraySize(rates) - 1)
            json += ",";
        
        json += "{";
        json += "\"time\":\"" + TimeToString(rates[i].time) + "\",";
        json += "\"open\":" + DoubleToString(rates[i].open, _Digits) + ",";
        json += "\"high\":" + DoubleToString(rates[i].high, _Digits) + ",";
        json += "\"low\":" + DoubleToString(rates[i].low, _Digits) + ",";
        json += "\"close\":" + DoubleToString(rates[i].close, _Digits) + ",";
        json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
        json += "}";
    }
    
    json += "]";
    return json;
}

//+------------------------------------------------------------------+
//| Parse prediction response                                        |
//+------------------------------------------------------------------+
bool ParsePredictionResponse(string response)
{
    // Simple JSON parsing (for production, use proper JSON library)
    int typePos = StringFind(response, "\"type\"");
    if(typePos < 0)
        return false;
    
    // Extract buy_prob
    int buyPos = StringFind(response, "\"buy_prob\"");
    if(buyPos >= 0)
    {
        int start = buyPos + 10;
        int end = StringFind(response, ",", start);
        if(end < 0) end = StringFind(response, "}", start);
        if(end > start)
        {
            string buyStr = StringSubstr(response, start, end - start);
            lastBuyProb = StringToDouble(buyStr);
        }
    }
    
    // Extract sell_prob
    int sellPos = StringFind(response, "\"sell_prob\"");
    if(sellPos >= 0)
    {
        int start = sellPos + 11;
        int end = StringFind(response, ",", start);
        if(end < 0) end = StringFind(response, "}", start);
        if(end > start)
        {
            string sellStr = StringSubstr(response, start, end - start);
            lastSellProb = StringToDouble(sellStr);
        }
    }
    
    // Extract regime
    int regimePos = StringFind(response, "\"regime\"");
    if(regimePos >= 0)
    {
        int start = StringFind(response, "\"", regimePos + 8) + 1;
        int end = StringFind(response, "\"", start);
        if(end > start)
        {
            lastRegime = StringSubstr(response, start, end - start);
        }
    }
    
    lastPrediction = response;
    
    // Print prediction
    Print("Prediction - Buy: ", DoubleToString(lastBuyProb, 3), 
          ", Sell: ", DoubleToString(lastSellProb, 3),
          ", Regime: ", lastRegime);
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic                                            |
//+------------------------------------------------------------------+
void ExecuteTradingLogic()
{
    // Check regime filter
    if(UseRegimeFilter && !IsRegimeAllowed(lastRegime))
    {
        return;
    }
    
    // Get current position
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
    if(lastBuyProb >= BuyThreshold && positionType != 0)
    {
        // Close sell position if exists
        if(positionType == 1)
        {
            trade.PositionClose(_Symbol);
        }
        
        // Open buy position
        if(!hasPosition || positionType == -1)
        {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double sl = StopLoss > 0 ? ask - StopLoss * _Point : 0;
            double tp = TakeProfit > 0 ? ask + TakeProfit * _Point : 0;
            
            if(trade.Buy(LotSize, _Symbol, ask, sl, tp, "Natron Buy Signal"))
            {
                Print("Buy order opened. Buy Prob: ", lastBuyProb);
            }
        }
    }
    else if(lastSellProb >= SellThreshold && positionType != 1)
    {
        // Close buy position if exists
        if(positionType == 0)
        {
            trade.PositionClose(_Symbol);
        }
        
        // Open sell position
        if(!hasPosition || positionType == -1)
        {
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double sl = StopLoss > 0 ? bid + StopLoss * _Point : 0;
            double tp = TakeProfit > 0 ? bid - TakeProfit * _Point : 0;
            
            if(trade.Sell(LotSize, _Symbol, bid, sl, tp, "Natron Sell Signal"))
            {
                Print("Sell order opened. Sell Prob: ", lastSellProb);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check if regime is allowed                                       |
//+------------------------------------------------------------------+
bool IsRegimeAllowed(string regime)
{
    string allowed[];
    int count = StringSplit(AllowedRegimes, ',', allowed);
    
    for(int i = 0; i < count; i++)
    {
        StringTrimLeft(allowed[i]);
        StringTrimRight(allowed[i]);
        if(StringCompare(allowed[i], regime, false) == 0)
            return true;
    }
    
    return false;
}

//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
    // Send ping to keep connection alive
    if(socketHandle != INVALID_HANDLE)
    {
        string ping = "{\"type\":\"ping\"}\n";
        SocketSend(socketHandle, ping);
    }
}
