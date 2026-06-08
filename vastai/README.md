# CRATTT Vast.ai Sweep Blocks

## Running Order
- Blocks V1-V12: Run on Vast.ai (one session at a time)
- Block V13: Run on Kaggle after all sessions complete

## Session Configuration (Block V1)
Change these three lines each session:
- Session 1: MODEL="dino", CORRUPTIONS="Weather+Digital"
- Session 2: MODEL="dino", CORRUPTIONS="Noise+Blur"  
- Session 3: MODEL="yolo", CORRUPTIONS="Weather+Digital"
- Session 4: MODEL="yolo", CORRUPTIONS="Noise+Blur"

## Important Notes
- Always use Block V6 corrected version (not original)
- Fog fix required in Block V10 from Session 2 onwards
- Upload previous session ZIP before running Block V10
- Download ZIP and destroy instance after each session
