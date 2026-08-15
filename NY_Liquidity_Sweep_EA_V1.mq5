//+------------------------------------------------------------------+
//|                                    NY_Liquidity_Sweep_EA_V1.mq5  |
//|                                                                    |
//|  HYPOTHESIS UNDER TEST:                                           |
//|  During the Asian session, XAUUSD frequently sweeps the           |
//|  previous New York session's High or Low, then reverses toward    |
//|  the opposite side.                                               |
//|                                                                    |
//|  This EA is a RESEARCH ENGINE FIRST, a trading robot second.      |
//|  Run it in Research Mode across years of history before ever      |
//|  enabling Trading Mode. The dashboard/statistics/CSV export        |
//|  exist so the hypothesis can be validated or rejected with        |
//|  evidence rather than assumed to be true.                         |
//|                                                                    |
//|  Compile in MetaEditor (F7) before use. Test in the Strategy      |
//|  Tester in Visual Mode first so you can see sweeps/BOS/trades      |
//|  drawn on the chart and sanity-check the logic.                    |
//+------------------------------------------------------------------+
#property copyright "Research & Execution EA - Educational Use"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//====================================================================
//  ENUMS
//====================================================================
enum ENUM_EA_MODE
  {
   MODE_RESEARCH = 0,   // Research Mode - log only, never trade
   MODE_TRADING  = 1    // Trading Mode - places live/demo trades
  };

enum ENUM_STRUCTURE_TYPE
  {
   STRUCT_BOS   = 0,    // Break of Structure
   STRUCT_CHOCH = 1     // Change of Character
  };

enum ENUM_ENTRY_TYPE
  {
   ENTRY_MARKET      = 0,
   ENTRY_LIMIT       = 1,
   ENTRY_RETRACEMENT = 2   // enter on retracement into sweep candle's body (50%)
  };

enum ENUM_SL_TYPE
  {
   SL_FIXED_POINTS = 0,
   SL_ATR          = 1,
   SL_ABOVE_SWEEP  = 2,   // for sells: above sweep high.  for buys: below sweep low.
   SL_STRUCTURE    = 3    // last opposing swing point
  };

enum ENUM_TP_TYPE
  {
   TP_1RR             = 0,
   TP_2RR             = 1,
   TP_3RR             = 2,
   TP_OPPOSITE_NY     = 3,
   TP_PREV_DAY_HIGH   = 4,
   TP_PREV_DAY_LOW    = 5,
   TP_ASIAN_HIGH      = 6,
   TP_ASIAN_LOW       = 7,
   TP_ATR_MULTIPLE    = 8
  };

enum ENUM_TRADE_DIR
  {
   DIR_NONE = 0,
   DIR_BUY  = 1,
   DIR_SELL = 2
  };

enum ENUM_SWEEP_SIDE
  {
   SWEEP_NONE = 0,
   SWEEP_HIGH = 1,
   SWEEP_LOW  = 2
  };

//====================================================================
//  INPUTS
//====================================================================
input group "===== MODE ====="
input ENUM_EA_MODE   InpMode              = MODE_RESEARCH;   // EA Mode
input string         InpCSVFileName       = "NYLiquiditySweep_Research.csv"; // Research CSV file name

input group "===== SESSIONS (Broker/Server Time, HH:MM) ====="
input string InpAsianStart   = "23:00";
input string InpAsianEnd     = "03:00";
input string InpLondonStart  = "06:00";
input string InpLondonEnd    = "10:15";
input string InpNYAMStart    = "12:00";
input string InpNYAMEnd      = "16:30";
input string InpNYPMStart    = "17:30";
input string InpNYPMEnd      = "20:30";

input group "===== SWEEP DETECTION ====="
input double InpMinSweepPoints   = 50;      // Minimum sweep size in points (5 pips-equivalent on 5-digit gold feeds; tune per broker)
input int    InpSwingLookback    = 5;       // Fractal lookback for swing highs/lows (BOS/CHoCH)
input ENUM_STRUCTURE_TYPE InpStructureRequired = STRUCT_BOS; // Confirmation required after sweep

input group "===== OPTIONAL FILTERS ====="
input bool   InpUseFVGFilter        = false;  // Require Fair Value Gap in direction of trade
input bool   InpUseATRFilter        = false;  // Require ATR above minimum (volatility filter)
input double InpMinATRPoints        = 200;    // Minimum ATR in points
input bool   InpUseVolumeFilter     = false;  // Require tick volume above average
input bool   InpUseMomentumCandle   = false;  // Require momentum candle on BOS bar
input double InpMomentumBodyPct     = 0.60;   // Min body/range ratio for momentum candle
input bool   InpUse3CandleFilter    = false;  // Require 3 consecutive same-direction candles
input bool   InpUsePrevDayBias      = false;  // Require previous day close direction to align
input bool   InpUseH4Bias           = false;  // Require H4 trend alignment (EMA20 slope)

input group "===== ENTRY ====="
input ENUM_ENTRY_TYPE InpEntryType   = ENTRY_MARKET;
input int    InpPendingExpirySec     = 3600;   // Pending order expiry (seconds), 0 = GTC

input group "===== STOP LOSS ====="
input ENUM_SL_TYPE InpSLType         = SL_ABOVE_SWEEP;
input double InpSLFixedPoints        = 300;
input double InpSLATRMultiple        = 1.5;
input double InpSLBufferPoints       = 30;    // extra buffer added beyond sweep/structure

input group "===== TAKE PROFIT ====="
input ENUM_TP_TYPE InpTPType         = TP_2RR;
input double InpTPATRMultiple        = 3.0;

input group "===== RISK MANAGEMENT ====="
input double InpRiskPercent          = 1.0;    // % of balance risked per trade
input double InpMaxDailyLossPercent  = 3.0;    // Max daily loss as % of balance, 0 = disabled
input int    InpMaxTradesPerDay      = 2;
input bool   InpUseBreakEven         = true;
input double InpBreakEvenTriggerRR   = 1.0;    // Move to BE once price reaches this RR
input bool   InpUseTrailingStop      = false;
input double InpTrailingStartRR      = 1.5;
input double InpTrailingStepPoints   = 100;
input bool   InpUsePartialTP         = false;
input double InpPartialTPPercent     = 50;     // % of position closed at 1RR
input double InpMaxSpreadPoints      = 500;
input int    InpMaxSlippagePoints    = 50;

input group "===== TRADING WINDOW ====="
input bool   InpTradeMonday          = true;
input bool   InpTradeTuesday         = true;
input bool   InpTradeWednesday       = true;
input bool   InpTradeThursday        = true;
input bool   InpTradeFriday          = true;

input group "===== DASHBOARD / VISUALS ====="
input bool   InpShowDashboard        = true;
input bool   InpDrawChartObjects     = true;
input int    InpDashboardX           = 15;
input int    InpDashboardY           = 25;

input group "===== ALERTS ====="
input bool   InpAlertPopup           = true;
input bool   InpAlertPush            = false;
input bool   InpAlertEmail           = false;
input bool   InpAlertSound           = true;
input string InpSoundFile            = "alert.wav";

input group "===== MAGIC ====="
input long   InpMagicNumber          = 20260702;
input string InpTradeComment         = "NYLiqSweepEA_V1";

//====================================================================
//  GLOBAL OBJECTS
//====================================================================
CTrade         trade;
CSymbolInfo    symbolInfo;

//====================================================================
//  STRUCTS
//====================================================================
struct SSessionWindow
  {
   int startMin;   // minutes-of-day, server time
   int endMin;
   bool overnight;
  };

struct SDayLevels
  {
   datetime date;          // midnight of the trading day (server time)
   double   nyHigh;
   double   nyLow;
   bool     nyComplete;    // NY PM session has closed, levels finalized
   double   asianHigh;
   double   asianLow;
   bool     nyHighFrozen;  // frozen at Asian session open for THIS day's Asian session
   double   frozenNYHigh;
   double   frozenNYLow;
   double   prevDayHigh;
   double   prevDayLow;
  };

struct SSweepState
  {
   bool             active;          // a sweep has occurred and we're waiting on structure/entry
   ENUM_SWEEP_SIDE  side;
   double           sweepPrice;      // extreme price reached during the sweep
   datetime         sweepTime;
   int              sweepBarShift;
   bool             structureConfirmed;
   ENUM_TRADE_DIR   direction;
   datetime         logDate;         // the trading day this sweep belongs to (for CSV correlation)
  };

struct SStats
  {
   int    totalSweeps;
   int    highSweeps;
   int    lowSweeps;
   int    winningSweeps;   // sweeps that led to a winning trade / correct reversal
   int    failedSweeps;
   int    bosCount;
   int    chochCount;
   int    tradesTotal;
   int    tradesWon;
   int    tradesLost;
   double grossProfit;
   double grossLoss;
   double sumRR;
   int    rrSamples;
   double bestDayPL;
   double worstDayPL;
  };

//====================================================================
//  GLOBAL STATE
//====================================================================
SSessionWindow g_asian, g_london, g_nyAM, g_nyPM;
SDayLevels     g_today;
SSweepState    g_sweep;
SStats         g_stats;

bool     g_inAsianLast   = false;
double   g_dailyStartBalance = 0;
int      g_tradesToday   = 0;
datetime g_lastTradeDay  = 0;
int      g_csvHandle     = INVALID_HANDLE;
string   g_dashboardPrefix = "NYLS_DASH_";
string   g_objPrefix       = "NYLS_OBJ_";
double   g_pipPoints       = 1.0; // adjust via point/digits below

//====================================================================
//  UTILITY: TIME PARSING
//====================================================================
int TimeStringToMinutes(const string hhmm)
  {
   string parts[];
   int n = StringSplit(hhmm, ':', parts);
   if(n < 2) return 0;
   int h = (int)StringToInteger(parts[0]);
   int m = (int)StringToInteger(parts[1]);
   return h*60+m;
  }

SSessionWindow BuildSession(const string startS, const string endS)
  {
   SSessionWindow w;
   w.startMin = TimeStringToMinutes(startS);
   w.endMin   = TimeStringToMinutes(endS);
   w.overnight = (w.endMin <= w.startMin);
   return w;
  }

int MinutesOfDay(const datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return dt.hour*60+dt.min;
  }

bool InSession(const datetime t, const SSessionWindow &w)
  {
   int mod = MinutesOfDay(t);
   if(!w.overnight)
      return (mod >= w.startMin && mod < w.endMin);
   else
      return (mod >= w.startMin || mod < w.endMin);
  }

datetime DayStart(const datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   return StructToTime(dt);
  }

//====================================================================
//  UTILITY: ALERTS  (Step 13)
//====================================================================
void RaiseAlert(const string msg)
  {
   string full = InpTradeComment + " | " + Symbol() + " | " + msg + " | " + TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES);
   if(InpAlertPopup)
      Alert(full);
   if(InpAlertSound)
      PlaySound(InpSoundFile);
   // Push notifications and email only make sense live - guard against tester spam/errors
   if(!MQLInfoInteger(MQL_TESTER))
     {
      if(InpAlertPush)
         SendNotification(full);
      if(InpAlertEmail)
         SendMail(InpTradeComment+" Alert", full);
     }
   Print(full);
  }

//====================================================================
//  RESEARCH LOGGER  (Step 16)
//====================================================================
bool OpenResearchLog()
  {
   int flags = FILE_CSV|FILE_WRITE|FILE_READ|FILE_ANSI|FILE_COMMON;
   bool exists = FileIsExist(InpCSVFileName, FILE_COMMON);
   g_csvHandle = FileOpen(InpCSVFileName, flags, ',');
   if(g_csvHandle == INVALID_HANDLE)
     {
      Print("Failed to open research CSV: ", GetLastError());
      return false;
     }
   FileSeek(g_csvHandle, 0, SEEK_END);
   if(!exists || FileSize(g_csvHandle) == 0)
     {
      FileWrite(g_csvHandle,
                "Date","Time","NYHigh","NYLow","AsianHigh","AsianLow",
                "HighSwept","LowSwept","SweepPrice","StructureConfirmed",
                "StructureType","Direction","Reversed","PipsMoved",
                "MFE","MAE","Result");
     }
   return true;
  }

void LogSweepRow(const SSweepState &s, double reversalPips, double mfe, double mae, const string result)
  {
   if(g_csvHandle == INVALID_HANDLE) return;
   FileSeek(g_csvHandle, 0, SEEK_END);
   FileWrite(g_csvHandle,
             TimeToString(s.logDate, TIME_DATE),
             TimeToString(s.sweepTime, TIME_MINUTES|TIME_SECONDS),
             DoubleToString(g_today.frozenNYHigh, Digits()),
             DoubleToString(g_today.frozenNYLow, Digits()),
             DoubleToString(g_today.asianHigh, Digits()),
             DoubleToString(g_today.asianLow, Digits()),
             (s.side==SWEEP_HIGH ? "TRUE":"FALSE"),
             (s.side==SWEEP_LOW  ? "TRUE":"FALSE"),
             DoubleToString(s.sweepPrice, Digits()),
             (s.structureConfirmed ? "TRUE":"FALSE"),
             (InpStructureRequired==STRUCT_BOS ? "BOS":"CHoCH"),
             (s.direction==DIR_BUY ? "BUY" : (s.direction==DIR_SELL ? "SELL":"NONE")),
             (reversalPips > 0 ? "TRUE":"FALSE"),
             DoubleToString(reversalPips, 1),
             DoubleToString(mfe, 1),
             DoubleToString(mae, 1),
             result);
  }

//====================================================================
//  SESSION / LEVEL TRACKING  (Step 1 & 2)
//====================================================================
// Computes the High/Low of a session window on the most recently
// completed occurrence of that session, using H1 bars for efficiency.
bool ComputeSessionRange(const SSessionWindow &w, const datetime dayAnchor, double &outHigh, double &outLow)
  {
   // Build the actual start/end datetimes around dayAnchor (midnight of the target day)
   datetime start = dayAnchor + w.startMin*60;
   datetime end;
   if(!w.overnight)
      end = dayAnchor + w.endMin*60;
   else
      end = dayAnchor + 24*3600 + w.endMin*60; // rolls into next day

   int barsTotal = Bars(Symbol(), PERIOD_M5);
   int startShift = iBarShift(Symbol(), PERIOD_M5, start, true);
   int endShift   = iBarShift(Symbol(), PERIOD_M5, end, true);
   if(startShift < 0 || endShift < 0) return false;
   if(startShift < endShift) return false; // not enough history yet

   double hi = -DBL_MAX, lo = DBL_MAX;
   bool got = false;
   for(int i = endShift; i <= startShift; i++)
     {
      datetime bt = iTime(Symbol(), PERIOD_M5, i);
      if(bt < start || bt >= end) continue;
      double h = iHigh(Symbol(), PERIOD_M5, i);
      double l = iLow(Symbol(), PERIOD_M5, i);
      if(h > hi) hi = h;
      if(l < lo) lo = l;
      got = true;
     }
   if(!got) return false;
   outHigh = hi; outLow = lo;
   return true;
  }

// Updates g_today: recomputes New York (AM+PM union) high/low for "today",
// tracks Asian high/low for the currently running Asian session, and
// freezes NY levels at the moment the Asian session begins (Step 2).
void UpdateDailyLevels()
  {
   datetime now = TimeCurrent();
   datetime today0 = DayStart(now);

   if(g_today.date != today0)
     {
      // New day rolled over - carry forward previous day's NY high/low as "prevDayHigh/Low"
      g_today.prevDayHigh = g_today.nyHigh;
      g_today.prevDayLow  = g_today.nyLow;
      g_today.date = today0;
      g_today.nyHigh = -DBL_MAX;
      g_today.nyLow  = DBL_MAX;
      g_today.nyComplete = false;
      g_today.asianHigh = -DBL_MAX;
      g_today.asianLow  = DBL_MAX;
      g_today.nyHighFrozen = false;
      g_inAsianLast = false;
      g_sweep.active = false;
      g_sweep.structureConfirmed = false;
     }

   // New York session union (AM start -> PM end) for "today" (server-day based on NY AM start)
   double nyH, nyL;
   SSessionWindow nyUnion;
   nyUnion.startMin = g_nyAM.startMin;
   nyUnion.endMin   = g_nyPM.endMin;
   nyUnion.overnight = (nyUnion.endMin <= nyUnion.startMin);
   if(ComputeSessionRange(nyUnion, today0, nyH, nyL))
     {
      g_today.nyHigh = nyH;
      g_today.nyLow  = nyL;
      if(!InSession(now, nyUnion))
         g_today.nyComplete = true;
     }

   // Asian session tracking - freeze NY levels on session start (Step 2)
   bool inAsian = InSession(now, g_asian);
   if(inAsian && !g_inAsianLast)
     {
      // Just entered Asian session -> freeze
      g_today.frozenNYHigh = g_today.nyHigh;
      g_today.frozenNYLow  = g_today.nyLow;
      g_today.nyHighFrozen = true;
      g_today.asianHigh = -DBL_MAX;
      g_today.asianLow  = DBL_MAX;
      g_sweep.active = false;
      g_sweep.structureConfirmed = false;
      RaiseAlert(StringFormat("Asian session started. Frozen NY High=%s Low=%s",
                 DoubleToString(g_today.frozenNYHigh, Digits()),
                 DoubleToString(g_today.frozenNYLow, Digits())));
     }
   if(inAsian)
     {
      double h = iHigh(Symbol(), PERIOD_M1, 0);
      double l = iLow(Symbol(), PERIOD_M1, 0);
      if(h > g_today.asianHigh) g_today.asianHigh = h;
      if(l < g_today.asianLow)  g_today.asianLow  = l;
     }
   g_inAsianLast = inAsian;
  }

//====================================================================
//  SWEEP DETECTION  (Step 3)
//====================================================================
// A sweep is: wick beyond the frozen NY level by at least InpMinSweepPoints,
// followed by a candle CLOSE back on the original side of that level.
ENUM_SWEEP_SIDE DetectSweep()
  {
   if(!g_today.nyHighFrozen) return SWEEP_NONE;
   if(!InSession(TimeCurrent(), g_asian)) return SWEEP_NONE;

   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double minDist = InpMinSweepPoints * point;

   double high1 = iHigh(Symbol(), PERIOD_M1, 1);
   double low1  = iLow(Symbol(), PERIOD_M1, 1);
   double close1= iClose(Symbol(), PERIOD_M1, 1);

   // Bullish sweep of the NY HIGH: wick exceeds NYHigh by min distance, closes back below it
   if(high1 >= g_today.frozenNYHigh + minDist && close1 < g_today.frozenNYHigh)
      return SWEEP_HIGH;

   // Bearish sweep of the NY LOW: wick breaks below NYLow by min distance, closes back above it
   if(low1 <= g_today.frozenNYLow - minDist && close1 > g_today.frozenNYLow)
      return SWEEP_LOW;

   return SWEEP_NONE;
  }

//====================================================================
//  MARKET STRUCTURE: SWING POINTS, BOS / CHoCH  (Step 4)
//====================================================================
// Simple fractal-based swing detection on M1: a swing high is a bar whose
// high is the highest among InpSwingLookback bars on either side; likewise
// for swing lows. BOS = close beyond the most recent swing point in the
// trend's direction. CHoCH = close beyond the most recent swing point
// AGAINST the prior trend direction (i.e. the first break that flips bias).
double FindLastSwingHigh(int startShift, int maxBars=300)
  {
   int n = InpSwingLookback;
   for(int i = startShift+n; i < startShift+maxBars; i++)
     {
      double h = iHigh(Symbol(), PERIOD_M1, i);
      bool isSwing = true;
      for(int j=1; j<=n; j++)
        {
         if(iHigh(Symbol(), PERIOD_M1, i-j) > h || iHigh(Symbol(), PERIOD_M1, i+j) > h)
           { isSwing = false; break; }
        }
      if(isSwing) return h;
     }
   return -1;
  }

double FindLastSwingLow(int startShift, int maxBars=300)
  {
   int n = InpSwingLookback;
   for(int i = startShift+n; i < startShift+maxBars; i++)
     {
      double l = iLow(Symbol(), PERIOD_M1, i);
      bool isSwing = true;
      for(int j=1; j<=n; j++)
        {
         if(iLow(Symbol(), PERIOD_M1, i-j) < l || iLow(Symbol(), PERIOD_M1, i+j) < l)
           { isSwing = false; break; }
        }
      if(isSwing) return l;
     }
   return -1;
  }

// Checks whether structure has confirmed a reversal in the expected direction
// following a sweep. Returns true and sets outType once confirmed.
bool CheckStructureConfirmation(ENUM_SWEEP_SIDE side, ENUM_STRUCTURE_TYPE &outType)
  {
   if(side == SWEEP_HIGH)
     {
      // Expect bearish structure break: close below the last swing low formed after the sweep
      double swingLow = FindLastSwingLow(1);
      if(swingLow <= 0) return false;
      double close1 = iClose(Symbol(), PERIOD_M1, 1);
      if(close1 < swingLow)
        {
         outType = (InpStructureRequired==STRUCT_CHOCH) ? STRUCT_CHOCH : STRUCT_BOS;
         return true;
        }
     }
   else if(side == SWEEP_LOW)
     {
      // Expect bullish structure break: close above the last swing high formed after the sweep
      double swingHigh = FindLastSwingHigh(1);
      if(swingHigh <= 0) return false;
      double close1 = iClose(Symbol(), PERIOD_M1, 1);
      if(close1 > swingHigh)
        {
         outType = (InpStructureRequired==STRUCT_CHOCH) ? STRUCT_CHOCH : STRUCT_BOS;
         return true;
        }
     }
   return false;
  }

//====================================================================
//  OPTIONAL FILTERS  (Step 5)
//====================================================================
bool FilterFVG(ENUM_TRADE_DIR dir)
  {
   // 3-candle Fair Value Gap check on M1: candle[2] high/low vs candle[0] low/high
   double h2 = iHigh(Symbol(), PERIOD_M1, 3);
   double l2 = iLow(Symbol(), PERIOD_M1, 3);
   double h0 = iHigh(Symbol(), PERIOD_M1, 1);
   double l0 = iLow(Symbol(), PERIOD_M1, 1);
   if(dir == DIR_BUY)
      return (l0 > h2); // bullish FVG: gap between candle[1] low and candle[3] high
   if(dir == DIR_SELL)
      return (h0 < l2); // bearish FVG
   return false;
  }

bool FilterATR()
  {
   int handle = iATR(Symbol(), PERIOD_M15, 14);
   if(handle == INVALID_HANDLE) return true; // fail open
   double buf[];
   if(CopyBuffer(handle, 0, 1, 1, buf) <= 0) return true;
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   return (buf[0] / point) >= InpMinATRPoints;
  }

bool FilterVolume()
  {
   long vols[];
   if(CopyTickVolume(Symbol(), PERIOD_M1, 1, 20, vols) <= 0) return true;
   long avg = 0;
   for(int i=1; i<ArraySize(vols); i++) avg += vols[i];
   if(ArraySize(vols) <= 1) return true;
   avg /= (ArraySize(vols)-1);
   return vols[0] >= avg;
  }

bool FilterMomentumCandle()
  {
   double o = iOpen(Symbol(), PERIOD_M1, 1);
   double c = iClose(Symbol(), PERIOD_M1, 1);
   double h = iHigh(Symbol(), PERIOD_M1, 1);
   double l = iLow(Symbol(), PERIOD_M1, 1);
   double range = h - l;
   if(range <= 0) return false;
   double body = MathAbs(c-o);
   return (body/range) >= InpMomentumBodyPct;
  }

bool Filter3Candles(ENUM_TRADE_DIR dir)
  {
   for(int i=1; i<=3; i++)
     {
      double o = iOpen(Symbol(), PERIOD_M1, i);
      double c = iClose(Symbol(), PERIOD_M1, i);
      if(dir==DIR_BUY  && c<=o) return false;
      if(dir==DIR_SELL && c>=o) return false;
     }
   return true;
  }

bool FilterPrevDayBias(ENUM_TRADE_DIR dir)
  {
   double prevOpen  = iOpen(Symbol(), PERIOD_D1, 1);
   double prevClose = iClose(Symbol(), PERIOD_D1, 1);
   if(dir==DIR_BUY)  return prevClose > prevOpen;
   if(dir==DIR_SELL) return prevClose < prevOpen;
   return true;
  }

bool FilterH4Bias(ENUM_TRADE_DIR dir)
  {
   int handle = iMA(Symbol(), PERIOD_H4, 20, 0, MODE_EMA, PRICE_CLOSE);
   if(handle == INVALID_HANDLE) return true;
   double buf[];
   if(CopyBuffer(handle, 0, 1, 3, buf) < 3) return true;
   double slope = buf[2] - buf[0];
   if(dir==DIR_BUY)  return slope >= 0;
   if(dir==DIR_SELL) return slope <= 0;
   return true;
  }

bool PassesAllFilters(ENUM_TRADE_DIR dir)
  {
   if(InpUseFVGFilter      && !FilterFVG(dir))            return false;
   if(InpUseATRFilter      && !FilterATR())                return false;
   if(InpUseVolumeFilter   && !FilterVolume())              return false;
   if(InpUseMomentumCandle && !FilterMomentumCandle())      return false;
   if(InpUse3CandleFilter  && !Filter3Candles(dir))         return false;
   if(InpUsePrevDayBias    && !FilterPrevDayBias(dir))      return false;
   if(InpUseH4Bias         && !FilterH4Bias(dir))           return false;
   return true;
  }

//====================================================================
//  RISK / LOT SIZING
//====================================================================
double CalcLotSize(double slDistancePoints)
  {
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskMoney = balance * (InpRiskPercent/100.0);
   double tickValue = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_SIZE);
   double point     = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   if(tickSize <= 0 || tickValue <= 0) return SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);

   double slDistPrice = slDistancePoints * point;
   double valuePerLot = (slDistPrice / tickSize) * tickValue;
   if(valuePerLot <= 0) return SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);

   double lots = riskMoney / valuePerLot;

   double minLot  = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_STEP);
   lots = MathFloor(lots/lotStep)*lotStep;
   lots = MathMax(minLot, MathMin(maxLot, lots));
   return lots;
  }

bool DailyLossLimitHit()
  {
   if(InpMaxDailyLossPercent <= 0) return false;
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   double lossPct = (g_dailyStartBalance - eq) / g_dailyStartBalance * 100.0;
   return lossPct >= InpMaxDailyLossPercent;
  }

bool WithinTradingDays()
  {
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   switch(dt.day_of_week)
     {
      case 1: return InpTradeMonday;
      case 2: return InpTradeTuesday;
      case 3: return InpTradeWednesday;
      case 4: return InpTradeThursday;
      case 5: return InpTradeFriday;
      default: return false; // no weekend trading
     }
  }

//====================================================================
//  ENTRY / SL / TP CALCULATION  (Steps 7-9)
//====================================================================
double GetATRValue()
  {
   int handle = iATR(Symbol(), PERIOD_M15, 14);
   if(handle == INVALID_HANDLE) return 0;
   double buf[];
   if(CopyBuffer(handle, 0, 1, 1, buf) <= 0) return 0;
   return buf[0];
  }

double CalcStopLoss(ENUM_TRADE_DIR dir, double entryPrice)
  {
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double buffer = InpSLBufferPoints * point;
   double sl = 0;
   switch(InpSLType)
     {
      case SL_FIXED_POINTS:
         sl = (dir==DIR_BUY) ? entryPrice - InpSLFixedPoints*point
                              : entryPrice + InpSLFixedPoints*point;
         break;
      case SL_ATR:
        {
         double atr = GetATRValue();
         sl = (dir==DIR_BUY) ? entryPrice - atr*InpSLATRMultiple
                              : entryPrice + atr*InpSLATRMultiple;
         break;
        }
      case SL_ABOVE_SWEEP:
         sl = (dir==DIR_BUY) ? g_sweep.sweepPrice - buffer
                              : g_sweep.sweepPrice + buffer;
         break;
      case SL_STRUCTURE:
        {
         if(dir==DIR_BUY)
           {
            double sw = FindLastSwingLow(1);
            sl = (sw>0) ? sw - buffer : entryPrice - InpSLFixedPoints*point;
           }
         else
           {
            double sw = FindLastSwingHigh(1);
            sl = (sw>0) ? sw + buffer : entryPrice + InpSLFixedPoints*point;
           }
         break;
        }
     }
   return sl;
  }

double CalcTakeProfit(ENUM_TRADE_DIR dir, double entryPrice, double slPrice)
  {
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double riskDist = MathAbs(entryPrice - slPrice);
   double tp = 0;
   switch(InpTPType)
     {
      case TP_1RR: tp = (dir==DIR_BUY) ? entryPrice + riskDist     : entryPrice - riskDist;     break;
      case TP_2RR: tp = (dir==DIR_BUY) ? entryPrice + riskDist*2.0 : entryPrice - riskDist*2.0;  break;
      case TP_3RR: tp = (dir==DIR_BUY) ? entryPrice + riskDist*3.0 : entryPrice - riskDist*3.0;  break;
      case TP_OPPOSITE_NY:
         tp = (dir==DIR_BUY) ? g_today.frozenNYHigh : g_today.frozenNYLow;
         break;
      case TP_PREV_DAY_HIGH: tp = g_today.prevDayHigh; break;
      case TP_PREV_DAY_LOW:  tp = g_today.prevDayLow;  break;
      case TP_ASIAN_HIGH:    tp = g_today.asianHigh;   break;
      case TP_ASIAN_LOW:     tp = g_today.asianLow;    break;
      case TP_ATR_MULTIPLE:
        {
         double atr = GetATRValue();
         tp = (dir==DIR_BUY) ? entryPrice + atr*InpTPATRMultiple : entryPrice - atr*InpTPATRMultiple;
         break;
        }
     }
   return tp;
  }

//====================================================================
//  TRADE EXECUTION  (Step 6, 7, 10)
//====================================================================
bool SpreadOK()
  {
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   long spread = SymbolInfoInteger(Symbol(), SYMBOL_SPREAD);
   return spread <= InpMaxSpreadPoints;
  }

void TryEnterTrade(ENUM_TRADE_DIR dir)
  {
   if(InpMode != MODE_TRADING) return;
   if(!WithinTradingDays()) return;
   if(DailyLossLimitHit())
     {
      RaiseAlert("Daily loss limit hit - no more trades today.");
      return;
     }
   if(g_tradesToday >= InpMaxTradesPerDay) return;
   if(!SpreadOK()) return;
   if(PositionsTotal() > 0) return; // one position at a time, keep it simple/safe

   symbolInfo.Name(Symbol());
   symbolInfo.RefreshRates();
   double ask = symbolInfo.Ask();
   double bid = symbolInfo.Bid();
   double entry = (dir==DIR_BUY) ? ask : bid;

   double sl = CalcStopLoss(dir, entry);
   double tp = CalcTakeProfit(dir, entry, sl);
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double slDistPoints = MathAbs(entry-sl)/point;
   if(slDistPoints <= 0) return;

   double lots = CalcLotSize(slDistPoints);
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpMaxSlippagePoints);

   bool ok = false;
   if(InpEntryType == ENTRY_MARKET)
     {
      ok = (dir==DIR_BUY) ? trade.Buy(lots, Symbol(), entry, sl, tp, InpTradeComment)
                           : trade.Sell(lots, Symbol(), entry, sl, tp, InpTradeComment);
     }
   else
     {
      // Limit / retracement entry: place a limit order at 50% retracement of the sweep candle
      double sweepHigh = iHigh(Symbol(), PERIOD_M1, g_sweep.sweepBarShift);
      double sweepLow  = iLow(Symbol(), PERIOD_M1, g_sweep.sweepBarShift);
      double mid = (sweepHigh+sweepLow)/2.0;
      datetime expiry = (InpPendingExpirySec>0) ? TimeCurrent()+InpPendingExpirySec : 0;
      if(dir==DIR_BUY)
         ok = trade.BuyLimit(lots, mid, Symbol(), sl, tp, ORDER_TIME_SPECIFIED, expiry, InpTradeComment);
      else
         ok = trade.SellLimit(lots, mid, Symbol(), sl, tp, ORDER_TIME_SPECIFIED, expiry, InpTradeComment);
     }

   if(ok)
     {
      g_tradesToday++;
      g_stats.tradesTotal++;
      RaiseAlert(StringFormat("Trade Executed: %s lots=%.2f entry=%s sl=%s tp=%s",
                 (dir==DIR_BUY?"BUY":"SELL"), lots,
                 DoubleToString(entry,Digits()), DoubleToString(sl,Digits()), DoubleToString(tp,Digits())));
      DrawTradeLevels(entry, sl, tp, dir);
     }
   else
     {
      Print("Trade failed: ", trade.ResultRetcodeDescription());
     }
  }

//====================================================================
//  POSITION MANAGEMENT: Break-even, trailing, partial TP  (Step 10)
//====================================================================
void ManageOpenPositions()
  {
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != Symbol()) continue;

      long   type   = PositionGetInteger(POSITION_TYPE);
      double open   = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl     = PositionGetDouble(POSITION_SL);
      double tp     = PositionGetDouble(POSITION_TP);
      double vol    = PositionGetDouble(POSITION_VOLUME);
      double price  = (type==POSITION_TYPE_BUY) ? symbolInfo.Bid() : symbolInfo.Ask();
      double point  = SymbolInfoDouble(Symbol(), SYMBOL_POINT);

      double riskDist = MathAbs(open - sl);
      if(riskDist <= 0) continue;
      double currentRR = (type==POSITION_TYPE_BUY) ? (price-open)/riskDist : (open-price)/riskDist;

      // Break-even
      if(InpUseBreakEven && currentRR >= InpBreakEvenTriggerRR)
        {
         bool needsMove = (type==POSITION_TYPE_BUY) ? (sl < open) : (sl > open);
         if(needsMove)
            trade.PositionModify(ticket, open, tp);
        }

      // Trailing stop
      if(InpUseTrailingStop && currentRR >= InpTrailingStartRR)
        {
         double newSL;
         if(type==POSITION_TYPE_BUY)
           {
            newSL = price - InpTrailingStepPoints*point;
            if(newSL > sl) trade.PositionModify(ticket, newSL, tp);
           }
         else
           {
            newSL = price + InpTrailingStepPoints*point;
            if(sl==0 || newSL < sl) trade.PositionModify(ticket, newSL, tp);
           }
        }

      // Partial take profit at 1RR
      if(InpUsePartialTP && currentRR >= 1.0)
        {
         double closeVol = NormalizeDouble(vol * (InpPartialTPPercent/100.0), 2);
         double minLot = SymbolInfoDouble(Symbol(), SYMBOL_VOLUME_MIN);
         if(closeVol >= minLot && closeVol < vol)
            trade.PositionClosePartial(ticket, closeVol);
        }
     }
  }

//====================================================================
//  CHART OBJECTS  (Step 12)
//====================================================================
void DrawLevelLine(const string name, double price, color clr, const string label)
  {
   if(!InpDrawChartObjects) return;
   string obj = g_objPrefix+name;
   if(ObjectFind(0, obj) < 0)
     {
      ObjectCreate(0, obj, OBJ_HLINE, 0, 0, price);
      ObjectSetInteger(0, obj, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, obj, OBJPROP_STYLE, STYLE_DOT);
      ObjectSetInteger(0, obj, OBJPROP_WIDTH, 1);
      ObjectSetString(0, obj, OBJPROP_TEXT, label);
     }
   else
     {
      ObjectSetDouble(0, obj, OBJPROP_PRICE, price);
     }
  }

void DrawTradeLevels(double entry, double sl, double tp, ENUM_TRADE_DIR dir)
  {
   if(!InpDrawChartObjects) return;
   DrawLevelLine("ENTRY", entry, clrWhite, "Entry");
   DrawLevelLine("SL", sl, clrRed, "Stop Loss");
   DrawLevelLine("TP", tp, clrLime, "Take Profit");
   string arrowName = g_objPrefix+"DIR_"+TimeToString(TimeCurrent());
   ObjectCreate(0, arrowName, (dir==DIR_BUY?OBJ_ARROW_BUY:OBJ_ARROW_SELL), 0, TimeCurrent(), entry);
   ObjectSetInteger(0, arrowName, OBJPROP_COLOR, (dir==DIR_BUY?clrLime:clrRed));
  }

void UpdateChartLevels()
  {
   if(!InpDrawChartObjects) return;
   if(g_today.nyHighFrozen)
     {
      DrawLevelLine("NYHIGH", g_today.frozenNYHigh, clrDodgerBlue, "Prev NY High");
      DrawLevelLine("NYLOW",  g_today.frozenNYLow,  clrDodgerBlue, "Prev NY Low");
     }
   if(g_today.asianHigh > -DBL_MAX/2)
      DrawLevelLine("ASIANHIGH", g_today.asianHigh, clrOrange, "Asian High");
   if(g_today.asianLow < DBL_MAX/2)
      DrawLevelLine("ASIANLOW", g_today.asianLow, clrOrange, "Asian Low");
  }

//====================================================================
//  DASHBOARD  (Step 11)
//====================================================================
void DashLabel(const string name, const string text, int x, int y, color clr, int fontSize=9)
  {
   string obj = g_dashboardPrefix+name;
   if(ObjectFind(0, obj) < 0)
     {
      ObjectCreate(0, obj, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, obj, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, obj, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, obj, OBJPROP_YDISTANCE, y);
      ObjectSetString(0, obj, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, obj, OBJPROP_FONTSIZE, fontSize);
     }
   ObjectSetString(0, obj, OBJPROP_TEXT, text);
   ObjectSetInteger(0, obj, OBJPROP_COLOR, clr);
  }

string CurrentSessionName()
  {
   datetime now = TimeCurrent();
   if(InSession(now, g_asian))  return "ASIAN";
   if(InSession(now, g_london)) return "LONDON";
   if(InSession(now, g_nyAM))   return "NEW YORK AM";
   if(InSession(now, g_nyPM))   return "NEW YORK PM";
   return "OFF-SESSION";
  }

void UpdateDashboard()
  {
   if(!InpShowDashboard) return;
   int x = InpDashboardX, y = InpDashboardY, lineH = 15, row=0;

   DashLabel("TITLE", "NY LIQUIDITY SWEEP EA  ["+(InpMode==MODE_RESEARCH?"RESEARCH":"TRADING")+"]", x, y+lineH*row++, clrGold, 10);
   DashLabel("SESSION", "Session: "+CurrentSessionName(), x, y+lineH*row++, clrWhite);
   DashLabel("NYH", StringFormat("NY High: %s", DoubleToString(g_today.frozenNYHigh,Digits())), x, y+lineH*row++, clrDodgerBlue);
   DashLabel("NYL", StringFormat("NY Low:  %s", DoubleToString(g_today.frozenNYLow,Digits())), x, y+lineH*row++, clrDodgerBlue);
   DashLabel("AH", StringFormat("Asian High: %s", DoubleToString(g_today.asianHigh,Digits())), x, y+lineH*row++, clrOrange);
   DashLabel("AL", StringFormat("Asian Low:  %s", DoubleToString(g_today.asianLow,Digits())), x, y+lineH*row++, clrOrange);
   DashLabel("SWEEP", "Sweep: "+(g_sweep.side==SWEEP_NONE?"none":(g_sweep.side==SWEEP_HIGH?"HIGH SWEPT":"LOW SWEPT")), x, y+lineH*row++, clrYellow);
   DashLabel("STRUCT", "Structure: "+(g_sweep.structureConfirmed?"CONFIRMED":"waiting"), x, y+lineH*row++, clrYellow);
   DashLabel("DIR", "Direction: "+(g_sweep.direction==DIR_BUY?"BUY":(g_sweep.direction==DIR_SELL?"SELL":"-")), x, y+lineH*row++, clrWhite);
   DashLabel("TRADESTODAY", StringFormat("Trades Today: %d / %d", g_tradesToday, InpMaxTradesPerDay), x, y+lineH*row++, clrWhite);

   double pl = AccountInfoDouble(ACCOUNT_EQUITY) - g_dailyStartBalance;
   DashLabel("PL", StringFormat("Today's P/L: %.2f", pl), x, y+lineH*row++, (pl>=0?clrLime:clrRed));

   int totalTrades = g_stats.tradesWon + g_stats.tradesLost;
   double winRate = totalTrades>0 ? (double)g_stats.tradesWon/totalTrades*100.0 : 0;
   double pf = (g_stats.grossLoss<0) ? g_stats.grossProfit/MathAbs(g_stats.grossLoss) : 0;
   DashLabel("WR", StringFormat("Win Rate: %.1f%%  (n=%d)", winRate, totalTrades), x, y+lineH*row++, clrWhite);
   DashLabel("PF", StringFormat("Profit Factor: %.2f", pf), x, y+lineH*row++, clrWhite);
   DashLabel("SWEEPSTATS", StringFormat("Sweeps: %d  (H:%d L:%d)  Win:%d Fail:%d",
             g_stats.totalSweeps, g_stats.highSweeps, g_stats.lowSweeps, g_stats.winningSweeps, g_stats.failedSweeps),
             x, y+lineH*row++, clrSilver);
  }

//====================================================================
//  MFE / MAE TRACKING FOR RESEARCH LOG
//====================================================================
double g_postSweepExtremeFav = 0, g_postSweepExtremeAdv = 0;
int    g_postSweepBarsCounted = 0;
const int POST_SWEEP_TRACK_BARS = 120; // ~2 hours on M1

void TrackPostSweepExcursion()
  {
   if(!g_sweep.active) return;
   double point = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   double close1 = iClose(Symbol(), PERIOD_M1, 1);
   double favMove, advMove;
   if(g_sweep.side == SWEEP_HIGH) // expecting reversal DOWN
     {
      favMove = (g_sweep.sweepPrice - iLow(Symbol(), PERIOD_M1, 1)) / point;
      advMove = (iHigh(Symbol(), PERIOD_M1, 1) - g_sweep.sweepPrice) / point;
     }
   else // SWEEP_LOW, expecting reversal UP
     {
      favMove = (iHigh(Symbol(), PERIOD_M1, 1) - g_sweep.sweepPrice) / point;
      advMove = (g_sweep.sweepPrice - iLow(Symbol(), PERIOD_M1, 1)) / point;
     }
   if(favMove > g_postSweepExtremeFav) g_postSweepExtremeFav = favMove;
   if(advMove > g_postSweepExtremeAdv) g_postSweepExtremeAdv = advMove;

   g_postSweepBarsCounted++;
   if(g_postSweepBarsCounted >= POST_SWEEP_TRACK_BARS)
     {
      double point2 = point;
      double reversalPips = g_postSweepExtremeFav;
      bool reversed = reversalPips > 0;
      if(reversed) g_stats.winningSweeps++; else g_stats.failedSweeps++;
      LogSweepRow(g_sweep, reversalPips, g_postSweepExtremeFav, g_postSweepExtremeAdv, reversed?"REVERSAL":"NO_REVERSAL");
      g_sweep.active = false;
      g_sweep.side = SWEEP_NONE;
      g_postSweepExtremeFav = 0; g_postSweepExtremeAdv = 0; g_postSweepBarsCounted = 0;
     }
  }

//====================================================================
//  CORE LOGIC  (Steps 3-6 orchestration)
//====================================================================
void CoreLogic()
  {
   UpdateDailyLevels();
   UpdateChartLevels();

   // Step 3: detect a new sweep if not already tracking one
   if(!g_sweep.active)
     {
      ENUM_SWEEP_SIDE side = DetectSweep();
      if(side != SWEEP_NONE)
        {
         g_sweep.active = true;
         g_sweep.side = side;
         g_sweep.sweepPrice = (side==SWEEP_HIGH) ? iHigh(Symbol(), PERIOD_M1, 1) : iLow(Symbol(), PERIOD_M1, 1);
         g_sweep.sweepTime = iTime(Symbol(), PERIOD_M1, 1);
         g_sweep.sweepBarShift = 1;
         g_sweep.structureConfirmed = false;
         g_sweep.direction = DIR_NONE;
         g_sweep.logDate = g_today.date;

         g_stats.totalSweeps++;
         if(side==SWEEP_HIGH) g_stats.highSweeps++; else g_stats.lowSweeps++;
         g_postSweepExtremeFav = 0; g_postSweepExtremeAdv = 0; g_postSweepBarsCounted = 0;

         RaiseAlert(side==SWEEP_HIGH ? "NY HIGH SWEPT - waiting BOS" : "NY LOW SWEPT - waiting BOS");
        }
     }

   // Step 4: if a sweep is active but not yet confirmed, look for structure break
   if(g_sweep.active && !g_sweep.structureConfirmed)
     {
      ENUM_STRUCTURE_TYPE stype;
      if(CheckStructureConfirmation(g_sweep.side, stype))
        {
         g_sweep.structureConfirmed = true;
         g_sweep.direction = (g_sweep.side==SWEEP_HIGH) ? DIR_SELL : DIR_BUY;
         if(stype==STRUCT_BOS) g_stats.bosCount++; else g_stats.chochCount++;

         RaiseAlert((stype==STRUCT_BOS?"Bearish/Bullish BOS Confirmed":"CHoCH Confirmed")+
                    " - "+(g_sweep.direction==DIR_BUY?"BUY READY":"SELL READY"));

         // Step 5: optional filters, then Step 6/7: enter trade (Trading Mode only)
         if(PassesAllFilters(g_sweep.direction))
            TryEnterTrade(g_sweep.direction);
        }
     }

   // Track MFE/MAE for research CSV regardless of mode
   TrackPostSweepExcursion();

   if(InpMode == MODE_TRADING)
      ManageOpenPositions();

   UpdateDashboard();
  }

//====================================================================
//  DAILY RESET
//====================================================================
void CheckDailyReset()
  {
   datetime today0 = DayStart(TimeCurrent());
   if(g_lastTradeDay != today0)
     {
      double pl = AccountInfoDouble(ACCOUNT_EQUITY) - g_dailyStartBalance;
      if(g_lastTradeDay != 0)
        {
         if(pl > g_stats.bestDayPL)  g_stats.bestDayPL  = pl;
         if(pl < g_stats.worstDayPL) g_stats.worstDayPL = pl;
        }
      g_lastTradeDay = today0;
      g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_tradesToday = 0;
     }
  }

//====================================================================
//  EXPERT LIFECYCLE
//====================================================================
int OnInit()
  {
   g_asian  = BuildSession(InpAsianStart, InpAsianEnd);
   g_london = BuildSession(InpLondonStart, InpLondonEnd);
   g_nyAM   = BuildSession(InpNYAMStart, InpNYAMEnd);
   g_nyPM   = BuildSession(InpNYPMStart, InpNYPMEnd);

   ZeroMemory(g_today);
   g_today.date = 0;
   ZeroMemory(g_sweep);
   ZeroMemory(g_stats);
   g_stats.worstDayPL = 0;

   g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_lastTradeDay = DayStart(TimeCurrent());
   g_tradesToday = 0;

   trade.SetExpertMagicNumber(InpMagicNumber);
   symbolInfo.Name(Symbol());

   if(!OpenResearchLog())
      Print("Warning: research CSV logging disabled.");

   EventSetTimer(1);
   Print("NY Liquidity Sweep EA V1 initialized. Mode=", (InpMode==MODE_RESEARCH?"RESEARCH":"TRADING"));
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_csvHandle != INVALID_HANDLE)
      FileClose(g_csvHandle);

   ObjectsDeleteAll(0, g_dashboardPrefix);
   ObjectsDeleteAll(0, g_objPrefix);

   PrintFormat("=== Session Summary === Sweeps:%d Wins:%d Fails:%d BOS:%d CHoCH:%d Trades:%d",
               g_stats.totalSweeps, g_stats.winningSweeps, g_stats.failedSweeps,
               g_stats.bosCount, g_stats.chochCount, g_stats.tradesTotal);
  }

void OnTick()
  {
   CheckDailyReset();
   CoreLogic();
  }

//+------------------------------------------------------------------+
//| Captures closed-deal results so the dashboard's Win Rate /        |
//| Profit Factor and OnTester's optimization score are accurate.     |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                         const MqlTradeRequest &request,
                         const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagicNumber) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return; // only closing deals

   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                  + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                  + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);

   if(profit >= 0)
     {
      g_stats.tradesWon++;
      g_stats.grossProfit += profit;
     }
   else
     {
      g_stats.tradesLost++;
      g_stats.grossLoss += profit;
     }
  }

void OnTimer()
  {
   // Keeps dashboard/session state fresh even without new ticks (e.g. thin overnight liquidity)
   CheckDailyReset();
   UpdateDailyLevels();
   UpdateDashboard();
  }

//+------------------------------------------------------------------+
//| OnTester - summarizes hypothesis-test results after a backtest    |
//| run. Access via the Strategy Tester "Optimization results" tab    |
//| or the Experts log for the printed summary above.                 |
//+------------------------------------------------------------------+
double OnTester()
  {
   int totalTrades = g_stats.tradesWon + g_stats.tradesLost;
   double winRate = totalTrades>0 ? (double)g_stats.tradesWon/totalTrades : 0;
   double pf = (g_stats.grossLoss<0) ? g_stats.grossProfit/MathAbs(g_stats.grossLoss) : 0;
   // Custom optimization criterion: reward high win-rate AND high profit factor together
   return winRate * pf;
  }
//+------------------------------------------------------------------+
