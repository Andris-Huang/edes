import sys
import os
import inspect
import importlib
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QAction, QTextCursor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QSplitter,
                             QListWidget, QTextEdit, QVBoxLayout, QLabel, QGroupBox,
                             QTabWidget, QFormLayout, QLineEdit, QScrollArea, QPushButton,
                             QFileDialog, QComboBox)

# ==============================================================================
# PIPELINE PRODUCTION IMPORT INTEGRATION
# ==============================================================================
try:
    import edes
    from edes.experiments import Experiment
    from edes.utils.utils import beep_python
    # Dynamically resolve base sequence objects directly from your backend environment
    edes_base = importlib.import_module("edes.experiments.sequences.base")
    BaseSequence = getattr(edes_base, "Sequence")
except ImportError as e:
    print(f"[FATAL DEPMISSING] Could not import edes package components: {e}")
    print("Ensure edes is installed or present in your PYTHONPATH.")
    sys.exit(1)


# ==============================================================================
# TERMINAL LIVE STREAM PROXY FOR TQDM WRITING CAPTURES
# ==============================================================================
class TerminalStreamProxy:
    """
    Simulates a file-like stream object to catch tqdm updates safely.
    Also implements __call__ so it can be executed directly as a logging function.
    """
    def __init__(self, text_edit_widget):
        self.widget = text_edit_widget
        self.last_line_was_pbar = False

    def __call__(self, message):
        """Allows the proxy object to be called directly like a function: proxy('text')"""
        self.write(str(message) + "\n")

    def write(self, text):
        if not text:
            return
        
        text_str = str(text)
        
        if text_str.startswith('\r'):
            text_str = text_str.lstrip('\r')
            if self.last_line_was_pbar:
                # Safe line clearing that avoids index out of bounds exceptions
                cursor = self.widget.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
            else:
                self.last_line_was_pbar = True
            
            cleaned = text_str.rstrip()
            if cleaned:
                self.widget.append(cleaned)
        else:
            cleaned_text = text_str.strip()
            if cleaned_text:
                self.widget.append(cleaned_text)
                self.last_line_was_pbar = False
        
        # Flush the UI updates immediately
        QApplication.processEvents()

    def flush(self):
        QApplication.processEvents()


# ==============================================================================
# MAIN GUI APPLICATION WINDOW
# ==============================================================================
class ExperimentControlGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electron Experimental Control Dashboard")
        self.resize(1500, 900) 

        # Memory Registries
        self.current_experiment = None  
        self.loaded_sequence_classes = {}
        self.active_param_inputs = {}  
        self.current_selected_sequence_name = None

        self.apply_theme()
        self.init_menu_bar()  

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)

        # --- LEFT COLUMN: Tab Workspace ---
        self.left_tab_widget = QTabWidget()
        
        self.tab_devices = QWidget()
        layout_devices_tab = QVBoxLayout(self.tab_devices)
        group_devices = QGroupBox("Connected Devices")
        layout_devices = QVBoxLayout()
        self.list_devices = QListWidget()
        self.list_devices.addItems(["[No Config Loaded - Use File Menu]"]) 
        layout_devices.addWidget(self.list_devices)
        group_devices.setLayout(layout_devices)
        layout_devices_tab.addWidget(group_devices)
        
        self.tab_parameters = QWidget()
        layout_params_tab = QVBoxLayout(self.tab_parameters)
        self.group_params = QGroupBox("Sequence Parameters: None Selected")
        self.form_layout = QFormLayout()
        self.form_layout.setSpacing(10)
        
        scroll_area = QScrollArea()
        scroll_container = QWidget()
        scroll_container.setLayout(self.form_layout)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_container)
        layout_params_tab.addWidget(scroll_area)
        
        self.btn_submit = QPushButton("Submit")
        self.btn_submit.setEnabled(False) 
        self.btn_submit.clicked.connect(self.on_submit_clicked)
        layout_params_tab.addWidget(self.btn_submit)
        
        self.left_tab_widget.addTab(self.tab_devices, "Devices")
        self.left_tab_widget.addTab(self.tab_parameters, "Parameters")
        self.main_splitter.addWidget(self.left_tab_widget)

        # --- RIGHT COLUMN ---
        group_plots = QGroupBox("Plotting Applets")
        layout_plots = QVBoxLayout()
        self.label_plots = QLabel("Live Plotting Canvas Placeholder")
        self.label_plots.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_plots.addWidget(self.label_plots)
        group_plots.setLayout(layout_plots)
        self.right_splitter.addWidget(group_plots)

        group_sequences = QGroupBox("Experimental Sequences")
        layout_sequences = QVBoxLayout()
        self.list_sequences = QListWidget()
        self.list_sequences.itemDoubleClicked.connect(self.on_sequence_double_clicked)
        layout_sequences.addWidget(self.list_sequences)
        group_sequences.setLayout(layout_sequences)
        self.right_splitter.addWidget(group_sequences)

        group_terminal = QGroupBox("Terminal Output")
        layout_terminal = QVBoxLayout()
        self.text_terminal = QTextEdit()
        self.text_terminal.setReadOnly(True)
        
        self.terminal_proxy = TerminalStreamProxy(self.text_terminal)
        self.terminal_proxy.write("[SYSTEM] GUI Initialized. Operational core loaded successfully.")
        layout_terminal.addWidget(self.text_terminal)
        group_terminal.setLayout(layout_terminal)
        self.right_splitter.addWidget(group_terminal)

        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setSizes([750, 750])
        self.right_splitter.setSizes([450, 220, 230])

        main_layout.addWidget(self.main_splitter)
        self.load_sequences()

    def init_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        load_config_action = QAction("Load Config", self)
        load_config_action.triggered.connect(self.on_load_config_clicked)
        file_menu.addAction(load_config_action)
        
        menubar.addMenu("Edit")
        menubar.addMenu("View")
        
        help_menu = menubar.addMenu("Help")
        solitary_action = QAction("You are on your own :P", self)
        solitary_action.triggered.connect(self.on_help_clicked)
        help_menu.addAction(solitary_action)

        menubar.setMouseTracking(True)
        menubar.installEventFilter(self)

    def eventFilter(self, source, event):
        if source is self.menuBar() and event.type() == QEvent.Type.MouseMove:
            menubar = self.menuBar()
            action = menubar.actionAt(event.position().toPoint())
            if action and action.menu() and not action.menu().isVisible():
                menubar.setActiveAction(action)
                action.menu().popup(menubar.mapToGlobal(menubar.actionGeometry(action).bottomLeft()))
                return True
        return super().eventFilter(source, event)

    def on_load_config_clicked(self):
        default_dir = "./experiments/configs"
        if not os.path.exists(default_dir):
            default_dir = ""

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration Python File", default_dir, "Python Files (*.py);;All Files (*)"
        )
        if not file_path:
            self.terminal_proxy.write("[WARNING] Configuration selection canceled.")
            return

        config_filename = os.path.splitext(os.path.basename(file_path))[0]
        self.terminal_proxy.write(f"\n[SYSTEM] Instantiating config: '{config_filename}'")
        
        try:
            if self.current_experiment:
                self.terminal_proxy.write("[SYSTEM] Tearing down active device connections...")
                self.current_experiment.close_all()

            # Pass proxy directly into the active experiment build layer
            self.current_experiment = Experiment(config_filename, log_callback=self.terminal_proxy)
            self.current_experiment.log_callback = self.terminal_proxy

            self.list_devices.clear()
            discovered_devices = self.current_experiment.list_devices()
            
            if discovered_devices:
                for dev in discovered_devices:
                    self.list_devices.addItem(f"• {dev}")
                self.terminal_proxy.write(f"[SUCCESS] Hardware mapping completed: {discovered_devices}")
            else:
                self.list_devices.addItem("[Config Loaded - No valid devices found]")
                self.terminal_proxy.write("[WARNING] Configuration mapped, but no instruments were initiated.")
                
            if self.current_selected_sequence_name:
                self.refresh_sequence_form()

        except Exception as e:
            self.terminal_proxy.write(f"[CRITICAL ABORT ERROR] Structural initialization failed: {str(e)}")

    def on_help_clicked(self):
        self.terminal_proxy.write("\n[HELP] System message: Just like what it says... you're on your own out here! God bless you 🚀")

    def on_sequence_double_clicked(self, item):
        self.current_selected_sequence_name = item.text()
        self.refresh_sequence_form()

    def refresh_sequence_form(self):
        if not self.current_selected_sequence_name: return
        seq_class = self.loaded_sequence_classes.get(self.current_selected_sequence_name)
        if not seq_class: return

        self.terminal_proxy.write(f"[SYSTEM] Generating form fields for {self.current_selected_sequence_name}")
        self.group_params.setTitle(f"Sequence Parameters: {self.current_selected_sequence_name}")

        self.active_param_inputs.clear()
        while self.form_layout.count() > 0:
            child = self.form_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        try:
            signature = inspect.signature(seq_class.__init__)
            for param_name, param in signature.parameters.items():
                if param_name in ['self', 'args', 'kwargs', 'log_callback'] or param.kind in [param.VAR_KEYWORD, param.VAR_POSITIONAL]:
                    continue
                
                is_none_default = (param.default is None)
                default_text = "" if (param.default is inspect.Parameter.empty or is_none_default) else str(param.default)

                # Route active experiment saving_dir properties automatically if present
                if param_name == "saving_dir" and self.current_experiment:
                    if hasattr(self.current_experiment, "saving_dir") and getattr(self.current_experiment, "saving_dir"):
                        default_text = str(self.current_experiment.saving_dir)
                        is_none_default = False

                if is_none_default:
                    input_widget = QComboBox()
                    input_widget.addItem("None", None)
                    if self.current_experiment:
                        for device_key in self.current_experiment.list_devices():
                            input_widget.addItem(device_key, device_key)
                    index = input_widget.findText(param_name)
                    if index >= 0: input_widget.setCurrentIndex(index)
                else:
                    input_widget = QLineEdit()
                    input_widget.setText(default_text)
                    if default_text == "":
                        input_widget.setPlaceholderText("Value required...")
                
                self.active_param_inputs[param_name] = input_widget
                self.form_layout.addRow(QLabel(f"{param_name}:"), input_widget)

            self.btn_submit.setEnabled(True)
            self.left_tab_widget.setCurrentIndex(1)

        except Exception as e:
            self.terminal_proxy.write(f"[ERROR] Signature mapping failure: {str(e)}")

    def _parse_input_value(self, text_val):
        cleaned = text_val.strip()
        if cleaned.lower() == 'none' or cleaned == '': return None
        if cleaned.lower() == 'true': return True
        if cleaned.lower() == 'false': return False
        try: return int(cleaned)
        except ValueError: pass
        try: return float(cleaned)
        except ValueError: pass
        return cleaned

    def on_submit_clicked(self):
        if not self.current_selected_sequence_name: return
        seq_class = self.loaded_sequence_classes.get(self.current_selected_sequence_name)
        if not seq_class: return

        sequence_kwargs = {}
        self.terminal_proxy.write("\n[SUBMIT] Processing arguments:")
        
        # --- LOG_CALLBACK PASS-THROUGH ROUTING ---
        # if 'log_callback' in inspect.signature(seq_class.__init__).parameters:
        if self.current_experiment and hasattr(self.current_experiment, 'log_callback') and getattr(self.current_experiment, 'log_callback'):
            sequence_kwargs['log_callback'] = self.current_experiment.log_callback
            self.terminal_proxy.write("  -> log_callback = Sourced from Experiment context handle")
        else:
            sequence_kwargs['log_callback'] = self.terminal_proxy
            self.terminal_proxy.write("  -> log_callback = Sourced from fallback GUI stream handler")

        for param_name, input_widget in self.active_param_inputs.items():
            if isinstance(input_widget, QComboBox):
                selected_data = input_widget.currentData()
                if selected_data is None:
                    sequence_kwargs[param_name] = None
                    self.terminal_proxy.write(f"  -> {param_name} = None")
                else:
                    if self.current_experiment and hasattr(self.current_experiment, selected_data):
                        sequence_kwargs[param_name] = getattr(self.current_experiment, selected_data)
                        self.terminal_proxy.write(f"  -> {param_name} = Linked Device Object Handle [{selected_data}]")
                    else:
                        sequence_kwargs[param_name] = None
                        self.terminal_proxy.write(f"  -> {param_name} = None (Device context reference missing)")
            else:
                raw_text = input_widget.text()
                parsed_value = self._parse_input_value(raw_text)
                sequence_kwargs[param_name] = parsed_value
                self.terminal_proxy.write(f"  -> {param_name} = {parsed_value} ({type(parsed_value).__name__})")

        try:
            self.terminal_proxy.write(f"[SYSTEM] Instantiating '{self.current_selected_sequence_name}' pipeline object...")
            sequence_instance = seq_class(**sequence_kwargs)
            self.terminal_proxy.write(f"[SYSTEM] Triggering sequence execution via run_save()...")
            results = sequence_instance.run_save()
            self.terminal_proxy.write(f"[SUCCESS] Execution completed.")
            beep_python()
        except Exception as e:
            self.terminal_proxy.write(f"[CRITICAL ATTEMPT ERROR] Processing failure: {str(e)}\n")

    def load_sequences(self):
        self.list_sequences.clear()
        self.loaded_sequence_classes.clear()
        try:
            for name, obj in inspect.getmembers(edes_base, inspect.isclass):
                if issubclass(obj, BaseSequence) and obj is not BaseSequence:
                    self.list_sequences.addItem(name)
                    self.loaded_sequence_classes[name] = obj
            self.terminal_proxy.write(f"[SYSTEM] Loaded {len(self.loaded_sequence_classes)} baseline sequence pipelines from library environment.")
        except Exception as e:
            self.terminal_proxy.write(f"[ERROR] Module load scan failure: {str(e)}")

    def closeEvent(self, event):
        if self.current_experiment:
            self.current_experiment.close_all()
        super().closeEvent(event)

    def apply_theme(self):
        theme = """
        QMainWindow { background-color: #E6EEF4; }
        QMenuBar { background-color: #1A365D; color: #FFFFFF; font-weight: bold; font-size: 11pt; padding: 2px; border-bottom: 2px solid #8FA8C1; }
        QMenuBar::item { background-color: transparent; padding: 6px 14px; margin: 2px 2px; }
        QMenuBar::item:selected { background-color: #D4AF37; color: #000000; border-radius: 4px; }
        QMenu { background-color: #FFFFFF; color: #000000; border: 2px solid #1A365D; border-radius: 4px; padding: 5px 0px; }
        QMenu::item { padding: 6px 30px 6px 20px; font-size: 10pt; }
        QMenu::item:selected { background-color: #D4AF37; color: #000000; }
        QGroupBox { background-color: #FFFFFF; border: 2px solid #8FA8C1; border-radius: 8px; margin-top: 1.5em; font-weight: bold; color: #1A365D; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 2px 10px; background-color: #1A365D; color: #FFFFFF; border-radius: 4px; }
        QTabWidget::pane { border: 2px solid #8FA8C1; border-radius: 8px; background-color: #FFFFFF; top: -2px; }
        QTabBar::tab { background-color: #CBD5E1; color: #1A365D; padding: 8px 20px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 4px; }
        QTabBar::tab:selected { background-color: #1A365D; color: #FFFFFF; }
        QTabBar::tab:hover:!selected { background-color: #B4C6D8; }
        QListWidget, QTextEdit, QScrollArea { background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 4px; color: #000000; padding: 5px; font-size: 11pt; font-family: 'Courier New', monospace; }
        QComboBox { background-color: #FFFFFF; border: 1px solid #8FA8C1; border-radius: 4px; padding: 5px; font-size: 11pt; color: #000000; }
        QComboBox:focus { border: 1px solid #D4AF37; }
        QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 20px; border-left: 1px solid #CBD5E1; }
        QComboBox QAbstractItemView { background-color: #FFFFFF; color: #000000; selection-background-color: #D4AF37; selection-color: #000000; border: 1px solid #8FA8C1; }
        QLineEdit { background-color: #FFFFFF; border: 1px solid #8FA8C1; border-radius: 4px; padding: 4px; font-size: 11pt; color: #000000; }
        QLineEdit:focus { border: 1px solid #D4AF37; }
        QPushButton { background-color: #1A365D; color: #FFFFFF; font-weight: bold; font-size: 12pt; border: none; border-radius: 6px; padding: 10px; margin-top: 5px; }
        QPushButton:hover { background-color: #2A4D7C; }
        QPushButton:pressed { background-color: #0F233F; }
        QPushButton:disabled { background-color: #CBD5E1; color: #64748B; }
        QListWidget::item:selected { background-color: #D4AF37; color: #000000; border-radius: 2px; }
        QListWidget::item:hover { background-color: #E2E8F0; }
        QSplitter::handle { background-color: #8FA8C1; margin: 2px; border-radius: 3px; }
        QSplitter::handle:horizontal { width: 6px; }
        QSplitter::handle:vertical { height: 6px; }
        """
        self.setStyleSheet(theme)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = ExperimentControlGUI()
    window.show()
    sys.exit(app.exec())