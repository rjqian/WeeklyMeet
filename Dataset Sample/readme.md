# Readme 

This folder contains some data samples. 

## Contents 

### bitstamp_btcusd_trade_records.csv 
Past 24 hours' *comprehensive* transaction records of Bitcoin, accessed via the Bitstamp Exchange API. 

Each row is one transaction record. It has five items (columns): 

**date**: The time this transaction happened. It is measured by the Unix system time. The time origin is 00:00:00, Jan. 1st, 1970. The time unit is second. 

**tid**: Transaction ID. A *unique* sequence of numbers to indicate this transaction. 

**price**: The price in this transaction. It is measured by USD. 

**type**: 0 for buy, 1 for sell.  

**amount**: The number of Bitcoin traded. 
