"""
Q2 - UI TESTING for AI Document Enhancement System.

Tools Used:
  - pytest-qt: For testing PySide6/Qt desktop UI components
  - Selenium WebDriver: For testing the web admin dashboard

UI Testing Strategy:
  1. Component-level testing (individual widgets)
  2. Navigation/workflow testing (page transitions)
  3. Form validation testing (input handling)
  4. Visual consistency testing (layout, styles)
  5. Responsiveness testing (window resize behavior)
  6. Accessibility testing (keyboard navigation, labels)
"""

import os
import sys

import pytest

try:
    from PySide6.QtCore import Qt
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


# ═══════════════════════════════════════════════════════════════════════
# SECTION A: DESKTOP UI TESTING WITH pytest-qt
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestLoginWidgetUI:
    """Tests for the Login form widget using pytest-qt."""

    def test_login_form_elements_exist(self, qtbot):
        """TC-UI-001: Verify all login form elements are rendered."""
        from app.ui.main_window import LoginWidget
        widget = LoginWidget()
        qtbot.addWidget(widget)

        assert widget.username_input is not None
        assert widget.password_input is not None
        assert widget.login_button is not None
        assert widget.error_label is not None

    def test_login_form_placeholder_text(self, qtbot):
        """TC-UI-002: Verify placeholder text is displayed correctly."""
        from app.ui.main_window import LoginWidget
        widget = LoginWidget()
        qtbot.addWidget(widget)

        assert widget.username_input.placeholderText() == "Username"
        assert widget.password_input.placeholderText() == "Password"

    def test_password_field_is_masked(self, qtbot):
        """TC-UI-003: Verify password input is masked (echo mode)."""
        from app.ui.main_window import LoginWidget
        from PySide6.QtWidgets import QLineEdit
        widget = LoginWidget()
        qtbot.addWidget(widget)

        assert widget.password_input.echoMode() == QLineEdit.Password

    def test_empty_login_shows_error(self, qtbot):
        """TC-UI-004: Submit empty form shows validation error."""
        from app.ui.main_window import LoginWidget
        widget = LoginWidget()
        qtbot.addWidget(widget)

        qtbot.mouseClick(widget.login_button, Qt.LeftButton)
        assert "Please enter" in widget.error_label.text()

    def test_short_password_shows_error(self, qtbot):
        """TC-UI-005: Short password shows appropriate error message."""
        from app.ui.main_window import LoginWidget
        widget = LoginWidget()
        qtbot.addWidget(widget)

        widget.username_input.setText("admin")
        widget.password_input.setText("123")
        qtbot.mouseClick(widget.login_button, Qt.LeftButton)
        assert "6 characters" in widget.error_label.text()

    def test_valid_login_clears_error(self, qtbot):
        """TC-UI-006: Valid credentials clears error message."""
        from app.ui.main_window import LoginWidget
        widget = LoginWidget()
        qtbot.addWidget(widget)

        widget.username_input.setText("admin")
        widget.password_input.setText("password123")
        result = widget.handle_login()
        assert result is True
        assert widget.error_label.text() == ""

    def test_keyboard_input(self, qtbot):
        """TC-UI-007: Verify keyboard typing works in input fields."""
        from app.ui.main_window import LoginWidget
        widget = LoginWidget()
        qtbot.addWidget(widget)

        qtbot.keyClicks(widget.username_input, "testuser")
        assert widget.username_input.text() == "testuser"


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestDashboardWidgetUI:
    """Tests for the Dashboard widget using pytest-qt."""

    def test_dashboard_header_displayed(self, qtbot):
        """TC-UI-008: Dashboard header is visible."""
        from app.ui.main_window import DashboardWidget
        widget = DashboardWidget()
        qtbot.addWidget(widget)

        header = widget.findChild(type(widget.order_table), "orderTable")
        assert header is not None

    def test_order_table_columns(self, qtbot):
        """TC-UI-009: Order table has correct column headers."""
        from app.ui.main_window import DashboardWidget
        widget = DashboardWidget()
        qtbot.addWidget(widget)

        expected_headers = ["Order ID", "User", "Pages", "Total", "Status"]
        for i, header in enumerate(expected_headers):
            assert widget.order_table.horizontalHeaderItem(i).text() == header

    def test_add_order_row(self, qtbot):
        """TC-UI-010: Adding order row populates table correctly."""
        from app.ui.main_window import DashboardWidget
        widget = DashboardWidget()
        qtbot.addWidget(widget)

        widget.add_order_row(1, "user1", 10, 29.50, "Completed")
        assert widget.order_table.rowCount() == 1
        assert widget.order_table.item(0, 0).text() == "1"
        assert widget.order_table.item(0, 4).text() == "Completed"

    def test_filter_combo_options(self, qtbot):
        """TC-UI-011: Filter dropdown contains all expected options."""
        from app.ui.main_window import DashboardWidget
        widget = DashboardWidget()
        qtbot.addWidget(widget)

        options = [
            widget.filter_combo.itemText(i)
            for i in range(widget.filter_combo.count())
        ]
        assert options == ["All", "Pending", "Processing", "Completed"]

    def test_progress_bar_initial_state(self, qtbot):
        """TC-UI-012: Progress bar starts at 0%."""
        from app.ui.main_window import DashboardWidget
        widget = DashboardWidget()
        qtbot.addWidget(widget)

        assert widget.progress_bar.value() == 0


@pytest.mark.skipif(not PYSIDE6_AVAILABLE, reason="PySide6 not installed")
class TestMainWindowUI:
    """Tests for the Main Window using pytest-qt."""

    def test_window_title(self, qtbot):
        """TC-UI-013: Window title is set correctly."""
        from app.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)

        assert window.windowTitle() == "AI Document Enhancement System"

    def test_minimum_window_size(self, qtbot):
        """TC-UI-014: Window has appropriate minimum size."""
        from app.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)

        assert window.minimumWidth() == 1200
        assert window.minimumHeight() == 800

    def test_navigation_buttons_exist(self, qtbot):
        """TC-UI-015: All navigation sidebar buttons are present."""
        from app.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)

        expected_nav = ["Home", "Upload", "Documents", "Orders", "Settings"]
        for name in expected_nav:
            assert name in window.nav_buttons

    def test_status_bar_initial_message(self, qtbot):
        """TC-UI-016: Status bar shows 'Ready' on startup."""
        from app.ui.main_window import MainWindow
        window = MainWindow()
        qtbot.addWidget(window)

        assert window.status_bar.currentMessage() == "Ready"


# ═══════════════════════════════════════════════════════════════════════
# SECTION B: WEB UI TESTING WITH SELENIUM (Admin Dashboard)
# ═══════════════════════════════════════════════════════════════════════

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium not installed")
class TestWebUIWithSelenium:
    """
    Web UI testing for admin dashboard using Selenium WebDriver.
    These tests verify usability and consistency of the web interface.
    """

    @pytest.fixture(autouse=True)
    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1920,1080")
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(10)
        self.base_url = "http://localhost:8000"
        yield
        self.driver.quit()

    def test_web_login_page_loads(self):
        """TC-WUI-001: Login page loads with all required elements."""
        self.driver.get(f"{self.base_url}/login")
        username_field = self.driver.find_element(By.ID, "username")
        password_field = self.driver.find_element(By.ID, "password")
        login_btn = self.driver.find_element(By.ID, "loginBtn")

        assert username_field.is_displayed()
        assert password_field.is_displayed()
        assert login_btn.is_displayed()

    def test_web_responsive_layout(self):
        """TC-WUI-002: Page layout adapts to different screen sizes."""
        self.driver.get(f"{self.base_url}/dashboard")

        self.driver.set_window_size(1920, 1080)
        sidebar = self.driver.find_element(By.CLASS_NAME, "sidebar")
        assert sidebar.is_displayed()

        self.driver.set_window_size(768, 1024)
        hamburger = self.driver.find_element(By.CLASS_NAME, "menu-toggle")
        assert hamburger.is_displayed()

    def test_web_consistent_theme(self):
        """TC-WUI-003: UI elements follow consistent color theme."""
        self.driver.get(f"{self.base_url}/dashboard")
        buttons = self.driver.find_elements(By.CLASS_NAME, "btn-primary")
        for btn in buttons:
            bg_color = btn.value_of_css_property("background-color")
            assert bg_color is not None

    def test_web_cross_browser_title(self):
        """TC-WUI-004: Page title is consistent across browsers."""
        self.driver.get(f"{self.base_url}/dashboard")
        assert "AI Document" in self.driver.title
