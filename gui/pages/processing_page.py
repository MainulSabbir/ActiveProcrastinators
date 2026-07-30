import tkinter as tk
from tkinter import ttk, messagebox

import threading, os
import traceback

# import tab frame creators directly
from gui.frames.process.execution_tab import create_execution_frame
from gui.frames.process.binarization_tab import create_binarization_frame
from gui.frames.process.flow_tab import create_flow_frame
from gui.frames.process.intensity_tab import create_intensity_frame
from gui.frames.process.nematics_tab import create_nematics_frame

from core import BarcodeConfig, InputConfig, PreviewConfig, AggregationConfig

# import configs directly
from gui.config import (
    BarcodeConfigGUI,
    InputConfigGUI,
    PreviewConfigGUI,
    AggregationConfigGUI,
)

from gui.window import setup_log_window, setup_results_window

def create_tabs(parent, config, input_config, preview_config):
    core_config = BarcodeConfig()
    notebook = ttk.Notebook(parent, takefocus=0)
    notebook.pack(fill="both", expand=True)

    execution_frame = create_execution_frame(notebook, config, input_config)
    binarization_frame = create_binarization_frame(notebook, config, preview_config, input_config)
    flow_frame = create_flow_frame(notebook, config, preview_config, input_config)
    intensity_frame = create_intensity_frame(notebook, config, preview_config, input_config)
    nematics_frame = create_nematics_frame(notebook, config, input_config)

    notebook.add(execution_frame, text="Execution Settings")
    notebook.add(binarization_frame, text="Binarization Settings")
    notebook.add(flow_frame, text="Optical Flow Settings")
    notebook.add(intensity_frame, text="Intensity Distribution Settings")
    notebook.add(nematics_frame, text="Nematics Analysis")

    return notebook

def create_processing_worker(
    config: BarcodeConfig,
    input_config: InputConfig,
    results_callback=None,
):
    """Create the worker function for processing in background thread.

    results_callback: optional callable(summary_dict) invoked when the
    Nematics branch produces a result summary (used to show it in the GUI).
    """

    if input_config.configuration_file:
        try:
            config = BarcodeConfig.load_from_yaml(input_config.configuration_file)
        except Exception as e:
            messagebox.showerror("Error reading config file", str(e))
            return

    def worker():
        try:
            mode = input_config.mode
            from core.pipeline import run_analysis

            # Handle file/directory processing
            file_path = input_config.file_path
            dir_path = input_config.dir_path

            if not (dir_path or file_path):
                messagebox.showerror(
                    "Error", "No file or directory has been selected."
                )
                return

            dir_name = dir_path if dir_path else file_path

            # Nematics branch runs independently of the standard BARCODE
            # pipeline (it reads images directly and needs no channel choice).
            if config.modules.nematics:
                from analysis import run_nematics_analysis
                summary = run_nematics_analysis(dir_name, config, input_config)
                if summary and results_callback:
                    results_callback(summary)

            standard_modules = (
                config.modules.image_binarization
                or config.modules.optical_flow
                or config.modules.intensity_distribution
            )

            # Run the standard pipeline unless this is a Nematics-only run.
            if standard_modules or not config.modules.nematics:
                channels = config.channels.parse_all_channels
                channel_selection = config.channels.selected_channel
                length_conversion = input_config.length
                time_conversion = input_config.time
                config.reader.um_pixel_ratio /= length_conversion
                config.reader.exposure_time /= time_conversion

                if not (channels or (channel_selection is not None)):
                    messagebox.showerror("Error", "No channel has been specified.")
                    return

                run_analysis(dir_name, config, input_config)

        except Exception as e:
            print(f"Error during processing: {e}")
            print(traceback.format_exc())
            messagebox.showerror("Error during processing", str(e))

        finally:
            messagebox.showinfo(
                "Processing Complete", "Analysis has finished successfully."
            )

    return worker

def create_process_page(parent, switch_page):
    frame = tk.Frame(parent)
    frame.pack(fill="both", expand=True)

    gui_config = BarcodeConfigGUI()
    gui_input_config = InputConfigGUI()
    gui_preview_config = PreviewConfigGUI()
    gui_aggregation_config = AggregationConfigGUI()

    def on_run():
        setup_log_window(parent)

        # force the value manually to see if that fixes it
        gui_input_config.mode.set("dir")

        # Convert GUI configs to pure data configs
        config = gui_config.config
        input_config = gui_input_config.config

        # Show the Nematics results in their own GUI window (marshalled onto
        # the main thread from the worker).
        def results_callback(summary):
            parent.after(0, lambda: setup_results_window(parent, summary))

        worker = create_processing_worker(config, input_config, results_callback)
        threading.Thread(target=worker, daemon=True).start()

    back_button = ttk.Button(frame, text="← Back", command=lambda: switch_page("home"))
    back_button.pack(anchor="w", padx=20, pady=10)

    create_tabs(
        frame,
        gui_config,
        gui_input_config,
        gui_preview_config,
    )

    run_button = ttk.Button(frame, text="Process Data", command=on_run)
    run_button.pack(pady=10)

    return frame