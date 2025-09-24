#property copyright "Grok by xAI"
#property link      "https://x.ai"
#property version   "1.00"
#property description "Simple EMA Crossover EA for EURUSD M1"
#property strict

#include <Trade\Trade.mqh>

// Inputs
input int FastEMAPeriod = 5;               // Fast EMA period
input int SlowEMAPeriod = 20;              // Slow EMA period
input double RiskPercent = 1.0;            // Risk % per trade
input int StopLossPips = 20;               // SL in pips
input int TakeProfitPips = 50;             // TP in pips
input int MaxSpreadPips = 3;               // Max allowed spread
input int MagicNumber = 12345;             // Unique identifier for trades

// Global variables
CTrade trade;
int fastEMAHandle;
int slowEMAHandle;

// OnInit function
int OnInit() {
   // Check if attached to correct symbol and timeframe
   if (_Symbol != "EURUSD" || _Period != PERIOD_M1) {
      Print("EA is designed for EURUSD M1 only.");
      return(INIT_FAILED);
   }
   
   // Create indicator handles
   fastEMAHandle = iMA(_Symbol, PERIOD_M1, FastEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   slowEMAHandle = iMA(_Symbol, PERIOD_M1, SlowEMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   
   if (fastEMAHandle == INVALID_HANDLE || slowEMAHandle == INVALID_HANDLE) {
      Print("Failed to create EMA handles.");
      return(INIT_FAILED);
   }
   
   trade.SetExpertMagicNumber(MagicNumber);
   return(INIT_SUCCEEDED);
}

// OnDeinit function
void OnDeinit(const int reason) {
   IndicatorRelease(fastEMAHandle);
   IndicatorRelease(slowEMAHandle);
}

// OnTick function
void OnTick() {
   // Check for sufficient bars
   if (Bars(_Symbol, PERIOD_M1) < SlowEMAPeriod) return;
   
   // Get EMA values for last two completed bars (shift 1 and 2)
   double fastEMA[2];
   double slowEMA[2];
   if (CopyBuffer(fastEMAHandle, 0, 1, 2, fastEMA) != 2 || CopyBuffer(slowEMAHandle, 0, 1, 2, slowEMA) != 2) return;
   
   // fastEMA[0] = older bar (shift 2), fastEMA[1] = recent bar (shift 1)
   bool buyCrossover = (fastEMA[0] <= slowEMA[0]) && (fastEMA[1] > slowEMA[1]);
   bool sellCrossover = (fastEMA[0] >= slowEMA[0]) && (fastEMA[1] < slowEMA[1]);
   
   // Check spread
   double spread = (Ask - Bid) / _Point;
   if (spread > MaxSpreadPips) return;
   
   // Check for open positions
   bool hasBuy = false;
   bool hasSell = false;
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber) {
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) hasBuy = true;
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) hasSell = true;
      }
   }
   
   // Close on reverse signal
   if (buyCrossover && hasSell) CloseAllSells();
   if (sellCrossover && hasBuy) CloseAllBuys();
   
   // Open new position if no open trades and signal
   if (!hasBuy && !hasSell) {
      double lotSize = CalculateLotSize();
      if (lotSize <= 0) return;
      
      if (buyCrossover) {
         double sl = Ask - StopLossPips * _Point * 10;  // Assuming 5-digit broker (pip = 10 points)
         double tp = Ask + TakeProfitPips * _Point * 10;
         trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, lotSize, 0, sl, tp, "Buy Crossover");
      }
      if (sellCrossover) {
         double sl = Bid + StopLossPips * _Point * 10;
         double tp = Bid - TakeProfitPips * _Point * 10;
         trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, lotSize, 0, sl, tp, "Sell Crossover");
      }
   }
}

// Function to calculate lot size based on risk
double CalculateLotSize() {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (RiskPercent / 100.0);
   double pipValue = 10.0;  // Standard for EURUSD per lot ($10 per pip)
   double slValue = StopLossPips * pipValue;
   double lots = NormalizeDouble(riskAmount / slValue, 2);
   if (lots < 0.01) lots = 0.01;  // Minimum lot
   return lots;
}

// Function to close all buy positions
void CloseAllBuys() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
         trade.PositionClose(PositionGetTicket(i));
      }
   }
}

// Function to close all sell positions
void CloseAllSells() {
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      if (PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == MagicNumber && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) {
         trade.PositionClose(PositionGetTicket(i));
      }
   }
}