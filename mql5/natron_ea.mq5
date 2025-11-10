//+------------------------------------------------------------------+
//|                                                      NatronEA.mq5 |
//|                                      Natron V2 AI Trading System |
//|                                                                   |
//+------------------------------------------------------------------+
#property copyright "Natron AI"
#property link      ""
#property version   "2.00"
#property description "AI-powered Expert Advisor using Natron Transformer model"

//--- Input parameters
input string   ServerIP = "127.0.0.1";          // Python server IP
input int      ServerPort = 9090;               // Python server port
input int      SequenceLength = 96;             // Number of candles to send
input double   MinConfidence = 0.6;             // Minimum confidence threshold
input double   MinBuyProb = 0.6;                // Minimum buy probability
input double   MinSellProb = 0.6;               // Minimum sell probability
input double   LotSize = 0.1;                   // Position size
input int      StopLoss = 100;                  // Stop loss in points
input int      TakeProfit = 200;                // Take profit in points
input int      MagicNumber = 12345;             // Magic number for orders
input bool     UseTrailingStop = true;          // Use trailing stop
input int      TrailingStop = 50;               // Trailing stop in points
input int      UpdateIntervalSeconds = 60;      // Update interval (seconds)

//--- Global variables
int socket_handle = INVALID_HANDLE;
datetime last_update_time = 0;
string last_regime = "";
double last_buy_prob = 0.0;
double last_sell_prob = 0.0;
double last_confidence = 0.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("Natron V2 EA Initialized");
    Print("Server: ", ServerIP, ":", ServerPort);
    Print("Sequence Length: ", SequenceLength);
    
    // Test connection
    if(!ConnectToServer())
    {
        Print("ERROR: Cannot connect to Python server!");
        Print("Please ensure the socket server is running.");
        return INIT_FAILED;
    }
    
    Print("✓ Successfully connected to Natron server");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(socket_handle != INVALID_HANDLE)
    {
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
    }
    
    Print("Natron V2 EA Deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    // Check if we should update (avoid spamming server)
    datetime current_time = TimeCurrent();
    if(current_time - last_update_time < UpdateIntervalSeconds)
        return;
    
    last_update_time = current_time;
    
    // Get prediction from AI model
    string prediction_json = GetAIPrediction();
    
    if(prediction_json == "")
    {
        Print("ERROR: Failed to get prediction from server");
        return;
    }
    
    // Parse prediction
    double buy_prob, sell_prob, confidence;
    string regime;
    int direction_class;
    
    if(!ParsePrediction(prediction_json, buy_prob, sell_prob, confidence, regime, direction_class))
    {
        Print("ERROR: Failed to parse prediction");
        return;
    }
    
    // Store for display
    last_buy_prob = buy_prob;
    last_sell_prob = sell_prob;
    last_confidence = confidence;
    last_regime = regime;
    
    // Display on chart
    DisplayInfo();
    
    // Check if confidence meets threshold
    if(confidence < MinConfidence)
    {
        Comment("Waiting for high confidence signal... (", DoubleToString(confidence, 2), ")");
        return;
    }
    
    // Execute trading logic
    ExecuteTradingLogic(buy_prob, sell_prob, confidence, regime);
    
    // Update trailing stops
    if(UseTrailingStop)
        UpdateTrailingStops();
}

//+------------------------------------------------------------------+
//| Connect to Python socket server                                    |
//+------------------------------------------------------------------+
bool ConnectToServer()
{
    if(socket_handle != INVALID_HANDLE)
    {
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
    }
    
    socket_handle = SocketCreate();
    
    if(socket_handle == INVALID_HANDLE)
    {
        Print("ERROR: Failed to create socket");
        return false;
    }
    
    // Try to connect
    if(!SocketConnect(socket_handle, ServerIP, ServerPort, 5000))
    {
        Print("ERROR: Failed to connect to ", ServerIP, ":", ServerPort);
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Get AI prediction from Python server                              |
//+------------------------------------------------------------------+
string GetAIPrediction()
{
    // Prepare OHLCV data (96 candles)
    string json_request = PrepareOHLCVData();
    
    if(json_request == "")
    {
        Print("ERROR: Failed to prepare OHLCV data");
        return "";
    }
    
    // Connect if not connected
    if(socket_handle == INVALID_HANDLE)
    {
        if(!ConnectToServer())
            return "";
    }
    
    // Send request
    json_request += "\n";  // Add newline terminator
    int sent = SocketSend(socket_handle, json_request);
    
    if(sent <= 0)
    {
        Print("ERROR: Failed to send data to server");
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
        return "";
    }
    
    // Receive response
    string response = "";
    uchar buffer[];
    ArrayResize(buffer, 4096);
    
    uint timeout = GetTickCount() + 5000;  // 5 second timeout
    
    while(GetTickCount() < timeout)
    {
        int received = SocketRead(socket_handle, buffer, 4096, 100);
        
        if(received > 0)
        {
            response += CharArrayToString(buffer, 0, received);
            
            // Check for newline terminator
            if(StringFind(response, "\n") >= 0)
                break;
        }
        else if(received < 0)
        {
            Print("ERROR: Socket read error");
            break;
        }
    }
    
    // Close connection (reconnect next time)
    SocketClose(socket_handle);
    socket_handle = INVALID_HANDLE;
    
    return response;
}

//+------------------------------------------------------------------+
//| Prepare OHLCV data as JSON                                        |
//+------------------------------------------------------------------+
string PrepareOHLCVData()
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    
    int copied = CopyRates(_Symbol, _Period, 0, SequenceLength, rates);
    
    if(copied != SequenceLength)
    {
        Print("ERROR: Failed to copy rates. Got ", copied, ", expected ", SequenceLength);
        return "";
    }
    
    // Build JSON
    string json = "{";
    json += "\"symbol\":\"" + _Symbol + "\",";
    json += "\"timeframe\":\"" + EnumToString(_Period) + "\",";
    json += "\"data\":[";
    
    for(int i = 0; i < SequenceLength; i++)
    {
        json += "[";
        json += IntegerToString(rates[i].time) + ",";
        json += DoubleToString(rates[i].open, _Digits) + ",";
        json += DoubleToString(rates[i].high, _Digits) + ",";
        json += DoubleToString(rates[i].low, _Digits) + ",";
        json += DoubleToString(rates[i].close, _Digits) + ",";
        json += IntegerToString(rates[i].tick_volume);
        json += "]";
        
        if(i < SequenceLength - 1)
            json += ",";
    }
    
    json += "]}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Parse prediction JSON                                              |
//+------------------------------------------------------------------+
bool ParsePrediction(string json, double &buy_prob, double &sell_prob, 
                    double &confidence, string &regime, int &direction_class)
{
    // Simple JSON parsing (you may want to use a proper JSON library)
    
    // Extract buy_prob
    int pos = StringFind(json, "\"buy_prob\":");
    if(pos >= 0)
    {
        string substr = StringSubstr(json, pos + 11);
        int end = StringFind(substr, ",");
        if(end < 0) end = StringFind(substr, "}");
        buy_prob = StringToDouble(StringSubstr(substr, 0, end));
    }
    
    // Extract sell_prob
    pos = StringFind(json, "\"sell_prob\":");
    if(pos >= 0)
    {
        string substr = StringSubstr(json, pos + 12);
        int end = StringFind(substr, ",");
        if(end < 0) end = StringFind(substr, "}");
        sell_prob = StringToDouble(StringSubstr(substr, 0, end));
    }
    
    // Extract confidence
    pos = StringFind(json, "\"confidence\":");
    if(pos >= 0)
    {
        string substr = StringSubstr(json, pos + 13);
        int end = StringFind(substr, ",");
        if(end < 0) end = StringFind(substr, "}");
        confidence = StringToDouble(StringSubstr(substr, 0, end));
    }
    
    // Extract regime
    pos = StringFind(json, "\"regime\":\"");
    if(pos >= 0)
    {
        string substr = StringSubstr(json, pos + 10);
        int end = StringFind(substr, "\"");
        regime = StringSubstr(substr, 0, end);
    }
    
    // Extract direction_class
    pos = StringFind(json, "\"direction_class\":");
    if(pos >= 0)
    {
        string substr = StringSubstr(json, pos + 18);
        int end = StringFind(substr, ",");
        if(end < 0) end = StringFind(substr, "}");
        direction_class = (int)StringToInteger(StringSubstr(substr, 0, end));
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute trading logic based on AI prediction                      |
//+------------------------------------------------------------------+
void ExecuteTradingLogic(double buy_prob, double sell_prob, double confidence, string regime)
{
    // Check if we already have an open position
    if(PositionsTotal() > 0)
    {
        // Manage existing position
        ManagePosition(buy_prob, sell_prob, confidence);
        return;
    }
    
    // Entry logic
    bool should_buy = (buy_prob >= MinBuyProb && buy_prob > sell_prob);
    bool should_sell = (sell_prob >= MinSellProb && sell_prob > buy_prob);
    
    // Avoid trading in highly volatile regimes (optional)
    if(regime == "VOLATILE")
    {
        Comment("Skipping trade - VOLATILE regime");
        return;
    }
    
    // Execute trades
    if(should_buy)
    {
        OpenPosition(ORDER_TYPE_BUY, buy_prob, confidence);
    }
    else if(should_sell)
    {
        OpenPosition(ORDER_TYPE_SELL, sell_prob, confidence);
    }
}

//+------------------------------------------------------------------+
//| Open a new position                                                |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE type, double probability, double confidence)
{
    double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) 
                                             : SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    double sl = 0, tp = 0;
    
    if(StopLoss > 0)
    {
        sl = (type == ORDER_TYPE_BUY) ? price - StopLoss * _Point 
                                       : price + StopLoss * _Point;
    }
    
    if(TakeProfit > 0)
    {
        tp = (type == ORDER_TYPE_BUY) ? price + TakeProfit * _Point 
                                       : price - TakeProfit * _Point;
    }
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = LotSize;
    request.type = type;
    request.price = price;
    request.sl = sl;
    request.tp = tp;
    request.deviation = 10;
    request.magic = MagicNumber;
    request.comment = "Natron AI (" + DoubleToString(probability, 2) + "/" + 
                      DoubleToString(confidence, 2) + ")";
    
    if(OrderSend(request, result))
    {
        Print("✓ ", (type == ORDER_TYPE_BUY ? "BUY" : "SELL"), 
              " order opened at ", price, 
              " | Prob: ", DoubleToString(probability, 2),
              " | Conf: ", DoubleToString(confidence, 2));
    }
    else
    {
        Print("ERROR: Failed to open order. Code: ", result.retcode);
    }
}

//+------------------------------------------------------------------+
//| Manage existing position                                           |
//+------------------------------------------------------------------+
void ManagePosition(double buy_prob, double sell_prob, double confidence)
{
    for(int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket <= 0) continue;
        
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        
        ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        
        // Close logic based on AI signal reversal
        bool should_close = false;
        
        if(pos_type == POSITION_TYPE_BUY && sell_prob > buy_prob + 0.2)
        {
            should_close = true;
            Print("AI signal reversal detected - closing BUY position");
        }
        else if(pos_type == POSITION_TYPE_SELL && buy_prob > sell_prob + 0.2)
        {
            should_close = true;
            Print("AI signal reversal detected - closing SELL position");
        }
        
        if(should_close)
        {
            ClosePosition(ticket);
        }
    }
}

//+------------------------------------------------------------------+
//| Close position                                                     |
//+------------------------------------------------------------------+
void ClosePosition(ulong ticket)
{
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action = TRADE_ACTION_DEAL;
    request.position = ticket;
    request.symbol = PositionGetString(POSITION_SYMBOL);
    request.volume = PositionGetDouble(POSITION_VOLUME);
    request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) 
                   ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    request.price = (request.type == ORDER_TYPE_SELL) 
                    ? SymbolInfoDouble(_Symbol, SYMBOL_BID) 
                    : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    request.deviation = 10;
    request.magic = MagicNumber;
    
    if(OrderSend(request, result))
    {
        Print("✓ Position closed: ", ticket);
    }
    else
    {
        Print("ERROR: Failed to close position. Code: ", result.retcode);
    }
}

//+------------------------------------------------------------------+
//| Update trailing stops                                              |
//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
    for(int i = 0; i < PositionsTotal(); i++)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket <= 0) continue;
        
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        
        double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
        double pos_sl = PositionGetDouble(POSITION_SL);
        ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        
        double current_price = (pos_type == POSITION_TYPE_BUY) 
                              ? SymbolInfoDouble(_Symbol, SYMBOL_BID) 
                              : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        
        double new_sl = 0;
        bool should_modify = false;
        
        if(pos_type == POSITION_TYPE_BUY)
        {
            new_sl = current_price - TrailingStop * _Point;
            if(new_sl > pos_sl && new_sl < current_price)
                should_modify = true;
        }
        else
        {
            new_sl = current_price + TrailingStop * _Point;
            if((new_sl < pos_sl || pos_sl == 0) && new_sl > current_price)
                should_modify = true;
        }
        
        if(should_modify)
        {
            MqlTradeRequest request = {};
            MqlTradeResult result = {};
            
            request.action = TRADE_ACTION_SLTP;
            request.position = ticket;
            request.symbol = _Symbol;
            request.sl = new_sl;
            request.tp = PositionGetDouble(POSITION_TP);
            
            OrderSend(request, result);
        }
    }
}

//+------------------------------------------------------------------+
//| Display info on chart                                              |
//+------------------------------------------------------------------+
void DisplayInfo()
{
    string info = "\n==================== NATRON V2 AI ====================\n";
    info += "Symbol: " + _Symbol + " | " + EnumToString(_Period) + "\n";
    info += "------------------------------------------------------\n";
    info += "Buy Probability:  " + DoubleToString(last_buy_prob * 100, 1) + "%\n";
    info += "Sell Probability: " + DoubleToString(last_sell_prob * 100, 1) + "%\n";
    info += "Confidence:       " + DoubleToString(last_confidence * 100, 1) + "%\n";
    info += "Market Regime:    " + last_regime + "\n";
    info += "------------------------------------------------------\n";
    info += "Open Positions: " + IntegerToString(PositionsTotal()) + "\n";
    info += "======================================================\n";
    
    Comment(info);
}

//+------------------------------------------------------------------+
