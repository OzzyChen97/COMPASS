"""Entry point for uniform text-only adaptation of HumanML component variants."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import train_humanml_independent_text as training
training.MODELS=('synonym','no_preserve','no_attribute','no_teacher','no_alter')
if __name__=='__main__':training.main()