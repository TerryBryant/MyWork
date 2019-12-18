@echo off
set cnt_mov=0
set cnt_mp4=0
for %%A in (*.MOV) do set /a cnt_mov+=1
for %%A in (*.mp4) do set /a cnt_mp4+=1
if %cnt_mov% gtr %cnt_mp4% (ren *.MOV *.mp4) else (ren *.mp4 *.MOV)
