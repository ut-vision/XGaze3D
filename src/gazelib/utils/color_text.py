class ColorText:
    """A simple text processor for printing colored text to the terminal."""
    
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m'
    }
    
    @classmethod
    def colorize(cls, text, color):
        """Colorize the given text using the specified color."""
        return f'{cls.colors[color]}{text}{cls.colors["reset"]}'
    
    @classmethod
    def black(cls, text):
        """Colorize the given text with black."""
        return cls.colorize(text, 'black')
    
    @classmethod
    def red(cls, text):
        """Colorize the given text with red."""
        return cls.colorize(text, 'red')
    
    @classmethod
    def green(cls, text):
        """Colorize the given text with green."""
        return cls.colorize(text, 'green')
    
    @classmethod
    def yellow(cls, text):
        """Colorize the given text with yellow."""
        return cls.colorize(text, 'yellow')
    
    @classmethod
    def blue(cls, text):
        """Colorize the given text with blue."""
        return cls.colorize(text, 'blue')
    
    @classmethod
    def magenta(cls, text):
        """Colorize the given text with magenta."""
        return cls.colorize(text, 'magenta')
    
    @classmethod
    def cyan(cls, text):
        """Colorize the given text with cyan."""
        return cls.colorize(text, 'cyan')
    
    @classmethod
    def white(cls, text):
        """Colorize the given text with white."""
        return cls.colorize(text, 'white')

def print_green(*args, **kwargs):
	out = ' '.join([str(arg) for arg in args])
	print(ColorText.green(out))
    
def print_yellow(*args, **kwargs):
	out = ' '.join([str(arg) for arg in args])
	print(ColorText.yellow(out))
def print_magenta(*args, **kwargs):
	out = ' '.join([str(arg) for arg in args])
	print(ColorText.magenta(out))
def print_cyan(*args, **kwargs):
	out = ' '.join([str(arg) for arg in args])
	print(ColorText.cyan(out))
def print_red(*args, **kwargs):
	out = ' '.join([str(arg) for arg in args])
	print(ColorText.red(out))

if __name__ == '__main__':
    print(ColorText.red('red'))
    print(ColorText.green('green'))
    print(ColorText.yellow('yellow'))
    print(ColorText.blue('blue'))
    print(ColorText.magenta('magenta'))
    print(ColorText.cyan('cyan'))
    print(ColorText.white('white'))
    print(ColorText.black('black'))