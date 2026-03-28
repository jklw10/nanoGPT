from parts.menu import PyGameMenu,  PygameDrawContext
import pygame
import json
import subprocess

def load_commands(filepath="commands.json"):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def make_command_runner(command_string):
    """
    Creates a function that runs the specific command when called by the menu.
    """
    def runner():
        print("\n--- Running Command ---")
        print(f"> {command_string}\n")
        subprocess.run(command_string, shell=True)
        print("\n--- Process Finished ---")
    return runner

def entry_creation_menu():
    print("\n--- Add New Command ---")
    name = input("Enter new command name (e.g., 'nanogpt mydataset'): ")
    if not name.strip():
        print("Aborted.")
        return
    cmd = input("Enter the command string: ")
    if not cmd.strip():
        print("Aborted.")
        return
    try:
        with open("commands.json", "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[name] = cmd
    with open("commands.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"[SUCCESS] Added '{name}'")

entries = load_commands("commands.json")
menu_items = []
for entry in entries:
    menu_items.append((entry, make_command_runner(entries[entry]))) 

menu_items.append(("Make new entry", entry_creation_menu))

pygame.init()
width = 800
height = 800
screen = pygame.display.set_mode((width, height))

pygame.display.set_caption("Select run")

clock  = pygame.time.Clock()
mainctx  = PygameDrawContext(screen, clock, 30, (0,0,0))
menurect = pygame.Rect(100, 100, width-100, height-100)

#print(menu_items)

menu = PyGameMenu(
    screen,
    menurect,
    spacing = 100,
    items=menu_items,
    draw_ctx=mainctx,
    on_exit=pygame.quit
)

menu.open_menu()  