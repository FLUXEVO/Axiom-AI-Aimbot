
import sys
import os

# Ensure src and bundled dependencies are in path
_src_dir = os.path.dirname(os.path.abspath(__file__))
_python_dir = os.path.join(_src_dir, "python")
_candidate_paths = [
    _src_dir,
    os.path.join(_python_dir, "dependencies"),
    os.path.join(_python_dir, "Lib", "site-packages"),
    os.path.join(_python_dir, "src", "python", "Lib", "site-packages"),
]
for _path in _candidate_paths:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from PyQt6.QtWidgets import QApplication
from gui.fluent_app.window import AxiomWindow
from core.config import Config, load_config, save_config
from core.config_manager import ConfigManager

_CONFIG_PATH = 'config.json'

def main():
    app = QApplication(sys.argv)

    # ── 判斷是否為全新安裝（config.json 尚未存在）──
    is_first_run = not os.path.exists(_CONFIG_PATH)

    # 初始化配置
    config = Config()
    load_config(config)

    # ── 首次啟動：顯示設置精靈 ──────────────────────
    if is_first_run:
        from gui.fluent_app.setup_wizard import SetupWizard
        wizard = SetupWizard(config)
        result = wizard.exec()
        # 無論用戶完成或關閉，套用主題並儲存設定
        config.dark_mode = wizard._isDark          # ← 同步到 config
        wizard.applyChosenTheme()
        save_config(config)

    # 初始化配置管理器
    cfg_manager = ConfigManager()

    # 建立視窗並注入配置（setConfig 會自動套用已保存的主題設定）
    window = AxiomWindow()
    window.setConfig(config)
    window.setConfigManager(cfg_manager)



    # 關閉時儲存配置
    app.aboutToQuit.connect(lambda: save_config(config))

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
