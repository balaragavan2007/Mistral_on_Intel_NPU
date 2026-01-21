import os
import time
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.rule import Rule
from rich.text import Text
from rich.align import Align
from rich.table import Table  # Added for metrics display

# CONFIG: NPU-optimized Channel-Wise model
MODEL_ID = "OpenVINO/mistral-7b-instruct-v0.3-int4-cw-ov"
MODEL_DIR = "mistral_model"
THEME_COLOR = "color(214)" 
console = Console()

class MistralCLI:
    def __init__(self):
        self.config = {"temp": 0.7, "tokens": 512, "device": "NPU"}
        self.chat_history = []
        self.perf_metrics = []  # List to store performance data
        self.pipe = None

    def print_logo(self):
        logo_text = r"""
  __  __ ___ ____ _____ ____      _    _     
 |  \/  |_ _/ ___|_   _|  _ \    / \  | |    
 | |\/| || |\___ \ | | | |_) |  / _ \ | |    
 | |  | || | ___) || | |  _ <  / ___ \| |    
 |_|  |_|___|____/ |_| |_| \_\/_/   \_\_|    
                                             
                 _    ___                       
                / \  |_ _|                     
               / _ \  | |                    
              / ___ \ | |                     
             /_/   \_\___|                     
"""
        styled_logo = Text(logo_text)
        styled_logo.stylize("bold color(226)", 0, 190)    
        styled_logo.stylize("bold color(214)", 191, 380)  
        styled_logo.stylize("bold color(202)", 381, 570)  
        styled_logo.stylize("bold color(196)", 571)       
        
        console.print(Align.center(styled_logo))
        console.print(Align.center(f"[{THEME_COLOR}]Running Locally on this pc[/{THEME_COLOR}]\n"))

    def show_metrics(self):
        """Displays a table of previous response speeds and latency."""
        if not self.perf_metrics:
            console.print(f"[{THEME_COLOR}]No performance data available yet.[/{THEME_COLOR}]")
            return

        table = Table(title="Performance History", border_style=THEME_COLOR)
        table.add_column("Query #", justify="center")
        table.add_column("Tokens", justify="center")
        table.add_column("Time (s)", justify="center")
        table.add_column("Speed (TPS)", justify="right", style="bold green")

        for i, m in enumerate(self.perf_metrics, 1):
            table.add_row(str(i), str(m['tokens']), f"{m['time']:.2f}", f"{m['tps']:.2f}")
        
        console.print(table)

    def setup(self):
        console.clear()
        if not os.path.exists(MODEL_DIR):
            console.print(f"[bold {THEME_COLOR}]✦[/bold {THEME_COLOR}] Downloading Local Weights...")
            snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
        
        prompt_text = f"[bold {THEME_COLOR}]Select Hardware[/bold {THEME_COLOR}] (NPU/GPU/CPU) [NPU]: "
        self.config['device'] = console.input(prompt_text).upper() or "NPU"
        
        console.print(f"[bold {THEME_COLOR}]✦[/bold {THEME_COLOR}] Booting Mistral on {self.config['device']}...")
        self.pipe = ov_genai.LLMPipeline(MODEL_DIR, self.config['device'])
        self.pipe.start_chat()
        
        console.clear()
        self.print_logo()
        console.print(Rule(f"Ready on {self.config['device']}", style=THEME_COLOR))
        console.print(Align.center(f"[{THEME_COLOR} dim]Type :help for commands[/{THEME_COLOR} dim]\n"))

    def handle_command(self, text):
        cmd = text.lower().strip()
        if cmd == ":exit": return "break"
        elif cmd == ":clear":
            self.pipe.finish_chat()
            self.pipe.start_chat()
            self.chat_history = []
            self.perf_metrics = []
            console.clear()
            self.print_logo()
            return "continue"
        elif cmd == ":metrics":
            self.show_metrics()
            return "continue"
        elif cmd == ":save":
            fname = f"chat_{int(time.time())}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write("\n".join(self.chat_history))
            console.print(f"[{THEME_COLOR}]✔ Saved to {fname}[/{THEME_COLOR}]")
            return "continue"
        elif cmd == ":help":
            console.print(Markdown("- `:metrics` - Show performance history\n- `:save` - Export chat\n- `:clear` - Reset history\n- `:exit` - Close"))
            return "continue"
        return None

    def run(self):
        while True:
            try:
                console.print("\n" + "─" * console.width, style=f"dim {THEME_COLOR}")
                user_input = console.input(f"[bold {THEME_COLOR}]You>[/bold {THEME_COLOR}] ").strip()
                
                if not user_input: continue
                action = self.handle_command(user_input)
                if action == "break": break
                if action == "continue": continue

                self.chat_history.append(f"You: {user_input}")
                console.print(f"\n[bold {THEME_COLOR} reverse] MISTRAL AI_ [/bold {THEME_COLOR} reverse]")
                
                full_response = ""
                token_count = 0
                start_time = time.time()

                with Live("", refresh_per_second=15) as live:
                    def streamer(token):
                        nonlocal full_response, token_count
                        full_response += token
                        token_count += 1
                        live.update(Markdown(full_response))
                        return False

                    self.pipe.generate(user_input, max_new_tokens=self.config['tokens'], streamer=streamer)
                
                # Calculate and store metrics
                end_time = time.time()
                duration = end_time - start_time
                tps = token_count / duration if duration > 0 else 0
                self.perf_metrics.append({"tokens": token_count, "time": duration, "tps": tps})

                console.print(f"[dim]Speed: {tps:.2f} TPS | Time: {duration:.2f}s[/dim]")
                self.chat_history.append(f"Mistral: {full_response}")

            except KeyboardInterrupt: break

if __name__ == "__main__":
    cli = MistralCLI()
    cli.setup()
    cli.run()