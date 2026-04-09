@echo off
python reset_commute_data.py
python collector.py commute
python auto_collect.py 5 288
pause
