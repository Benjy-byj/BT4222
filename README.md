# BT4222

## Files structure:
### /code/
- data_analysis.ipynb: Jupyter Notebook for data analysis and visualization.
- randomforest_baseline.ipynb: Jupyter Notebook for building a Random Forest using original dataset as baseline model.
- randomforest_with_feature_engineering.ipynb: Jupyter Notebook for building a Random Forest model with manually conducted feature engineering, such as Total Production Value, Production Concentration Index, etc.
- randomforest_gnn.ipynb: Jupyter Notebook for building a Random Forest model with features generated from Graph Neural Networks (GNN) (as well as manually conducted feature engineering).
- CatBoost_baseline.ipynb: Jupyter Notebook for building a CatBoost using original dataset as baseline model.
- CatBoost_with_feature_engineering.ipynb: Jupyter Notebook for building a CatBoost model with manually conducted feature engineering.
- CatBoost_gnn.ipynb: Jupyter Notebook for building a CatBoost model with features generated from Graph Neural Networks (GNN) (as well as manually conducted feature engineering).
### /data/
- Main_Data_Bank.xlsx: The initial raw data.
- 01222_20250925-055717.xlsx: This concerns data on Norway's population, obtained from https://www.ssb.no/en/statbank/table/01222/tableViewLayout1/. The timeframe spans from Q1 2010 to Q2 2025.
- API_NY.GDP.MKTP.CD_DS2_en_csv_v2_1011502: Historical data on Norway's GDP, obtained from https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=NO.
- Economic trends. Forecasts.csv： Norwegian economic data forecasts, including Gross Domestic Product (GDP), domestic GDP, employment figures, unemployment rate (level), standard annual wage, Consumer Price Index (CPI), etc., are presented as year-on-year percentage changes unless otherwise specified. Data source: https://www.ssb.no/en/nasjonalregnskap-og-konjunkturer/konjunkturer/statistikk/konjunkturtendensene 
- Number of animals per holding keeping various kinds of animal, by county and size of holding.csv: Data source: https://www.ssb.no/en/jord-skog-jakt-og-fiskeri/jordbruk/statistikk/gardsbruk-jordbruksareal-og-husdyr in table 2.
- Domestic animals.csv: Data source: https://www.ssb.no/en/jord-skog-jakt-og-fiskeri/jordbruk/statistikk/gardsbruk-jordbruksareal-og-husdyr.
- Agricultural area by use. Decares.csv: Data source: https://www.ssb.no/en/jord-skog-jakt-og-fiskeri/jordbruk/statistikk/gardsbruk-jordbruksareal-og-husdyr.
- 04181_20250925-062327.xlsx: Carcasses approved for human consumption (tonnes), by contents, region, carcasses approved and half year. Data source: https://www.ssb.no/en/statbank/table/04181/tableViewLayout1/.
- 14154_20250925-062754.xlsx: Pesticide use. Data source: https://www.ssb.no/en/statbank/table/14154/tableViewLayout1/.
- Yield per decare of potatoes and meadows for mowing, by county. Kilos.csv: Yield per decare of potatoes and meadows for mowing, by county. Kilos. Data source: https://www.ssb.no/en/jord-skog-jakt-og-fiskeri/jordbruk/statistikk/potet-og-grovforavlingar . 
- 12462_20250925-063448.xlsx: Producer price index. Oil/ gas extraction, manufacturing, mining and electricity (2021=100), by market, industry/commodity group, month and contents. Data source: https://www.ssb.no/en/statbank/table/12462/tableViewLayout1/. 
- 04984_20250925-063820.xlsx: Entrepreneurial income from agriculture for holders, by contents, region and year. Data source: https://www.ssb.no/en/statbank/table/04984/tableViewLayout1/. 
- FAOSTAT_data_en_9-25-2025.csv: Crops and livestock products. Data source: https://www.fao.org/faostat/en/#data/QCL
- Consumer Price Indices.csv: Data source: https://www.fao.org/faostat/en/#data/CP.
- Temperature change on land.csv: Temperature change on land: Data source: https://www.fao.org/faostat/en/#data/ET.
- Emissions totals .csv: Data source: https://www.fao.org/faostat/en/#data/GT.
- FAOSTAT_data_en_9-25-2025 (2).csv: Employment Indicators: Agriculture and agrifood systems. Data source: https://www.fao.org/faostat/en/#data/OEA.
- Government Expenditure.csv: investment Government Expenditure.csv. Data sources: https://www.fao.org/faostat/en/#data.

### /tools/
- merge_to_workbook.py: Combine all CSV and XLSX files into a single sheet and save as one XLSX file.
    ```
    pip install pandas openpyxl xlsxwriter xlrd

    python3 merge_to_workbook.py --input ./../data --output merged_main_sheet.xlsx --all-xlsx-sheets
    ```
