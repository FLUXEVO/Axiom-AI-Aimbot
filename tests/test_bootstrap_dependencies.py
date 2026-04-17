from unittest.mock import patch


def test_main_succeeds_without_pip_if_core_modules_exist():
    from tools import bootstrap_dependencies as bd

    with patch.object(bd, "_missing_core_modules", return_value=[]), \
        patch.object(bd, "_ensure_pip", return_value=1), \
        patch.object(bd, "_download_all_backend_wheels") as mock_download:
        result = bd.main()

    assert result == 0
    mock_download.assert_not_called()


def test_main_fails_without_pip_if_modules_missing():
    from tools import bootstrap_dependencies as bd

    with patch.object(bd, "_missing_core_modules", return_value=["numpy"]), \
        patch.object(bd, "_ensure_pip", return_value=1), \
        patch.object(bd, "_install_default_runtime") as mock_install:
        result = bd.main()

    assert result == 1
    mock_install.assert_not_called()


def test_main_treats_wheel_download_failure_as_non_fatal():
    from tools import bootstrap_dependencies as bd

    with patch.object(bd, "_missing_core_modules", return_value=[]), \
        patch.object(bd, "_ensure_pip", return_value=0), \
        patch.object(bd, "_download_all_backend_wheels", return_value=1):
        result = bd.main()

    assert result == 0

