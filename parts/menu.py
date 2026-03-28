from contextlib import nullcontext
from enum import Enum

from time import time
import pygame

class MenuAction(Enum):
	none    = 0
	up      = 1
	down    = 2
	enter   = 3
	exit    = 4
	jump    = 5

class Menu():
	"""
	 A menu base class you can make a menu tree easily from.
      see PyGameMenu for reference
	"""	    
	def __init__(self, menu_items, draw_f, input_f, on_exit=None, draw_ctx=None):
		self.menu_items = menu_items
		self.selected 	= 0
		self.draw_ctx	= draw_ctx if draw_ctx is not None else nullcontext()
		self.on_exit    = on_exit
		self.draw       = draw_f
		self.input      = input_f

	def exit():
		return MenuAction.exit
	
	def open_menu(self):
		while True: 
			self._draw_menu()  
			should_exit = self._get_menu_input()
			if should_exit == MenuAction.exit:
				if self.on_exit:
					self.on_exit()
				break
	
	def _draw_menu(self):
		with self.draw_ctx:
			for idx, item in enumerate(self.menu_items):
				self.draw(item[0], idx, self.selected) 
	
	def _get_menu_input(self):
		x = self.input()
		menu_length = max(1, len(self.menu_items))
		self.selected = min(self.selected, menu_length - 1)
		action = x[0]
		index  = x[1]
		match action:
			case MenuAction.up:
				self.selected = (self.selected - 1) % menu_length
			case MenuAction.down:
				self.selected = (self.selected + 1) % menu_length
			case MenuAction.enter:
				if index is not None:
					self.selected = index
				selected_function = self.menu_items[self.selected][1]
				function_result = selected_function()
				return function_result
			case MenuAction.jump: 
				self.selected = index % menu_length
			case _:
				return
			

class DrawMode(Enum):
    vertical    = 0
    horizontal  = 1

class Button:
    def __init__(self, 
                 rect_size,
                 text, 
                 draggable      = False,
                 font           = None, 
                 color          = (100, 100, 100), 
                 hover_color    = (150, 150, 150), 
                 text_color     = (255, 255, 255), 
                 ):
        self.text = text
        if font is None:
            font = pygame.font.SysFont("Arial", 24)
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.rect = pygame.Rect(rect_size) 
        self.draggable = draggable

        # State Queue tracking
        self.hovered        = False
        self.is_pressed     = False
        self.is_dragging    = False
        self._pending_click = False 
        self.drag_offset    = (0, 0)
        self.press_begin    = None

    def draw(self, surface, layout_rect, is_hovered=False):
        if self.is_dragging:
            layout_rect = self.rect
        else:
            self.rect = layout_rect
        
        current_color = self.hover_color if is_hovered else self.color
        pygame.draw.rect(surface, current_color, layout_rect)
        pygame.draw.rect(surface, (0, 0, 0), layout_rect, 2)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=layout_rect.center)
        surface.blit(text_surf, text_rect)
    
    def process_mouse_state(self, event):
        match event.type:
            case pygame.MOUSEBUTTONDOWN:
                if event.button != 1:
                    return False
                if not self.rect.collidepoint(event.pos):
                    return False
                self.is_pressed = True
                self.is_dragging = False
                self._pending_click = True 
                self.drag_offset = (self.rect.x - event.pos[0], self.rect.y - event.pos[1])
                self.press_begin = time()
            case pygame.MOUSEBUTTONUP:
                if event.button != 1:
                    return False
                if not self.is_pressed:
                    return False
                was_dragging = self.is_dragging
                self.is_pressed = False
                self.is_dragging = False
                return was_dragging
            case pygame.MOUSEMOTION:
                self.hovered = False
                if self.rect and self.rect.collidepoint(event.pos):
                    self.hovered = True
                if not self.is_pressed:
                    return False 
                if (self.press_begin - time()) < 0.1:
                    return False 
                self.is_dragging = True
                self._pending_click = False 
                self.rect.x = event.pos[0] + self.drag_offset[0]
                self.rect.y = event.pos[1] + self.drag_offset[1]
            case _:
                return False
        return True
        
    def is_clicked(self, event, rect =None):
        if not rect:
            if not self.rect:
                return Exception("No Collision box associated, or given to button click event check")
            rect = self.rect
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False
    
    def is_hovered(self, event):
        if event and event.type == pygame.MOUSEMOTION:
            if self.rect and self.rect.collidepoint(event.pos):
                return True
        return False

class InputScheme():
    """
    Defines how Pygame events translate to abstract MenuActions.
    Attributes:
        per_item (dict): Maps a string method name to a MenuAction. 
                         PRIORITY IS IMPLICIT based on dictionary insertion order!
                         Put high-priority interactions (like 'is_clicked') first.
        per_key (dict):  Maps pygame key constants to a MenuAction.
    """
    def __init__(self, per_item=None, per_key=None):
        self.per_item = per_item
        self.per_key = per_key

class PygameDrawContext:
    def __init__(self, screen, clock=None, fps=60, bg_color=None):
        self.screen = screen
        self.clock = clock
        self.fps = fps
        self.bg_color = bg_color
    def __enter__(self):
        if self.bg_color:
            self.screen.fill(self.bg_color)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.fps)
        return False
    
class PyGameMenu(Menu):
    """
    Creates a menu using Pygame as the drawer.	
    Example usage:
    
    .. code-block:: python	
    	screen = pygame.display.set_mode((common.SCREEN_WIDTH, common.SCREEN_HEIGHT))
    	clock = pygame.time.Clock()
    	mainctx = PygameDrawContext(screen, clock, common.FPS, common.BLACK)
    	menurect = pygame.Rect(100, 100, common.SCREEN_WIDTH - 100, common.SCREEN_HEIGHT - 100)    
    	menu = PyGameMenu(
    		screen,
    		menurect,
    		spacing=100,
    		items={
    		    ("New Run", main),
    		    ("Settings", settings),
    		    ("Quit", Menu.exit),
    		},
    		draw_ctx=mainctx,
    		on_exit=pygame.quit
            add_exit = False
    	)
    	menu.open_menu()  
    """    
    def __init__(self,
                surface :pygame.Surface,
                rect    :pygame.Rect,
                spacing :pygame.Rect,
                items,
                constructor  = Button,
                draw_mode    = DrawMode.vertical,
                cons_args    = pygame.Rect(0,0,500,100),
                on_exit      = None,
                input_scheme = None,
		        draw_ctx	 = None,
                add_exit     = True
            ):
        self.surface     = surface
        self.start_x     = rect.left
        self.start_y     = rect.top
        self.spacing     = spacing
        self.draw_mode   = draw_mode
        self.constructor = constructor
        self.cons_args   = cons_args

        super().__init__([], self._draw, self._input, on_exit, draw_ctx)
        for key, func in items:
            self.add_menu_item(key, func)
            
        if add_exit:
            self.add_menu_item("exit", Menu.exit)

        if input_scheme is None:
            self.input_scheme = InputScheme(per_key={
                pygame.K_w: MenuAction.up,
                pygame.K_s: MenuAction.down,
                pygame.K_RETURN: MenuAction.enter,
                pygame.K_ESCAPE: MenuAction.exit,
            },
            per_item={
                #"process_mouse_state":  MenuAction.none,
                "is_clicked":           MenuAction.enter,
                "is_hovered":           MenuAction.jump,
                })
        else:
            self.input_scheme = input_scheme
    
    def add_menu_item(self, key, func):
        if isinstance(key, str):
            key = self.constructor(self.cons_args, key)
        self.menu_items.append((key, func))

    def _draw(self, button, idx, selected_idx):
        layout_rect = pygame.Rect(self.start_x, self.start_y, button.rect.w, button.rect.h)
        if self.draw_mode == DrawMode.vertical:
            layout_rect.top += (idx * self.spacing)
        else:
            layout_rect.left += (idx * self.spacing)
        
        button.draw(self.surface, layout_rect, is_hovered=(idx == selected_idx))

    def _consume_event(self, event):     
        if event.type == pygame.QUIT:
            return True, MenuAction.exit, None
        for idx, (button, _) in enumerate(self.menu_items):
            for handle in self.input_scheme.per_item:
                func = getattr(button, handle, None)
                if func and func(event):
                    return True, self.input_scheme.per_item[handle], idx
        if event.type == pygame.KEYDOWN:
            if event.key in self.input_scheme.per_key:
                return True, self.input_scheme.per_key[event.key], None
                
        return False, MenuAction.none, None
    
    def _input(self):
        action_out = MenuAction.none
        action_idx = None

        for event in pygame.event.get():
            consumed, action, idx = self._consume_event(event)
            
            if consumed and action != MenuAction.none and action_out == MenuAction.none:
                action_out = action
                action_idx = idx

        return (action_out, action_idx)
