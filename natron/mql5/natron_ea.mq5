//+------------------------------------------------------------------+
//|                                                  natron_ea.mq5   |
//|                        Natron AI Trading System - MQL5 EA       |
//+------------------------------------------------------------------+
#property copyright "Natron Trading System"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- Input parameters
input int      SocketPort = 8888;           // Socket port
input string   ServerHost = "localhost";    // Server host
input int      MagicNumber = 234000;        // Magic number
input double   RiskPercent = 2.0;           // Risk per trade (%)
input int      MaxDailyTrades = 10;         // Max daily trades

//--- Global variables
CTrade trade;
datetime lastBarTime = 0;
int socketHandle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(20);
   trade.SetTypeFilling(ORDER_FILLING_IOC);
   
   Print("Natron EA initialized");
   Print("Connecting to Natron Server at ", ServerHost, ":", SocketPort);
   
   // Initialize socket connection (simplified - in production use proper socket library)
   // Note: MQL5 doesn't have built-in socket support, this is a placeholder
   // In production, use DLL or external library for socket communication
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(socketHandle != INVALID_HANDLE)
   {
      // Close socket
   }
   
   Print("Natron EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);
   
   if(currentBarTime != lastBarTime)
   {
      lastBarTime = currentBarTime;
      
      // Send candle data to Natron Server
      SendCandleData();
      
      // Receive predictions and execute trades
      ProcessPredictions();
   }
   
   // Manage existing positions (trailing stop, etc.)
   ManagePositions();
}

//+------------------------------------------------------------------+
//| Send candle data to Natron Server                                |
//+------------------------------------------------------------------+
void SendCandleData()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   if(CopyRates(_Symbol, PERIOD_CURRENT, 0, 1, rates) <= 0)
   {
      Print("Failed to get rates");
      return;
   }
   
   // Prepare candle data
   string candleData = StringFormat(
      "{\"type\":\"candle\",\"data\":{"
      "\"time\":\"%s\","
      "\"open\":%.5f,"
      "\"high\":%.5f,"
      "\"low\":%.5f,"
      "\"close\":%.5f,"
      "\"volume\":%lld,"
      "\"atr\":%.5f"
      "}}\n",
      TimeToString(rates[0].time, TIME_DATE|TIME_SECONDS),
      rates[0].open,
      rates[0].high,
      rates[0].low,
      rates[0].close,
      rates[0].tick_volume,
      iATR(_Symbol, PERIOD_CURRENT, 14, 0)
   );
   
   // Send to socket (placeholder - implement actual socket send)
   // SocketSend(socketHandle, candleData);
   
   Print("Sent candle data: ", candleData);
}

//+------------------------------------------------------------------+
//| Process predictions from Natron Server                           |
//+------------------------------------------------------------------+
void ProcessPredictions()
{
   // Receive prediction from socket (placeholder)
   // string predictionJson = SocketReceive(socketHandle);
   
   // Parse prediction and execute trades
   // In production, parse JSON and execute based on prediction
}

//+------------------------------------------------------------------+
//| Manage existing positions                                         |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      
      if(ticket <= 0)
         continue;
      
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;
      
      // Implement trailing stop logic here
      // Check ATR-based trailing stop conditions
      
      double positionOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL = PositionGetDouble(POSITION_SL);
      double currentTP = PositionGetDouble(POSITION_TP);
      
      // Trailing stop logic (simplified)
      double atr = iATR(_Symbol, PERIOD_CURRENT, 14, 0);
      double trailingDistance = atr * 1.5;
      
      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
      {
         double newSL = SymbolInfoDouble(_Symbol, SYMBOL_BID) - trailingDistance;
         if(newSL > currentSL && newSL > positionOpenPrice)
         {
            trade.PositionModify(ticket, newSL, currentTP);
         }
      }
      else if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL)
      {
         double newSL = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + trailingDistance;
         if((currentSL == 0 || newSL < currentSL) && newSL < positionOpenPrice)
         {
            trade.PositionModify(ticket, newSL, currentTP);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if daily trade limit reached                               |
//+------------------------------------------------------------------+
bool CheckDailyTradeLimit()
{
   int dailyTrades = 0;
   datetime todayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
   
   HistorySelect(todayStart, TimeCurrent());
   
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = HistoryDealGetTicket(i);
      
      if(ticket == 0)
         continue;
      
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol)
         continue;
      
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != MagicNumber)
         continue;
      
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN)
      {
         dailyTrades++;
      }
   }
   
   return dailyTrades >= MaxDailyTrades;
}

//+------------------------------------------------------------------+
