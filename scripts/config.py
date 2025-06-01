class Config:
    INTENT_MODEL = "phi4-mini-reasoning"
    CODE_MODEL = "deepseek-coder-v2"
    CHAT_MODEL = "deepseek-coder"
    
    TIMEOUT = 30
    MAX_RETRIES = 2
    
    MEMORY_LIMIT_MB = 128
    
    ENABLE_AUTO_FIX = True
    ENABLE_MEMORY_LIMIT = True
    
    SHOW_GENERATED_CODE = True
    SHOW_EXECUTION_STEPS = True
    SHOW_DEBUG_INFO = False
    
    SAFE_MODE = True
    ALLOWED_IMPORTS = [
        'math', 'random', 'datetime', 'json', 'csv', 're', 
        'collections', 'itertools', 'functools', 'operator',
        'tkinter', 'pygame', 'PIL', 'numpy', 'pandas', 'matplotlib'
    ]
    
    MODELS = {
        'fast': 'deepseek-r1:1.5b',
        'balanced': 'deepseek-r1:7b',
        'powerful': 'deepseek-r1',
        'large': 'deepseek-r1:32b',
        'massive': 'deepseek-r1:671b',
        'coder': 'deepseek-coder',
        'chat': 'llama3.1',
    }
    
    @classmethod
    def use_model_preset(cls, preset: str):
        if preset in cls.MODELS:
            model = cls.MODELS[preset]
            cls.INTENT_MODEL = model
            cls.CODE_MODEL = model
            cls.CHAT_MODEL = model
            print(f"🔄 Switched to {preset} model: {model}")
        else:
            print(f"❌ Unknown preset: {preset}")
            print(f"Available presets: {list(cls.MODELS.keys())}")
    
    @classmethod
    def use_custom_models(cls, intent_model=None, code_model=None, chat_model=None):
        if intent_model:
            cls.INTENT_MODEL = intent_model
        if code_model:
            cls.CODE_MODEL = code_model
        if chat_model:
            cls.CHAT_MODEL = chat_model
        print(f"🔄 Updated models - Intent: {cls.INTENT_MODEL}, Code: {cls.CODE_MODEL}, Chat: {cls.CHAT_MODEL}")
    
    @classmethod
    def set_performance_mode(cls, mode: str):
        if mode == "fast":
            cls.TIMEOUT = 15
            cls.MAX_RETRIES = 1
            cls.SHOW_GENERATED_CODE = False
            cls.use_model_preset("fast")
        elif mode == "balanced":
            cls.TIMEOUT = 30
            cls.MAX_RETRIES = 2
            cls.SHOW_GENERATED_CODE = True
            cls.use_model_preset("balanced")
        elif mode == "quality":
            cls.TIMEOUT = 60
            cls.MAX_RETRIES = 3
            cls.SHOW_GENERATED_CODE = True
            cls.use_model_preset("powerful")
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: fast, balanced, quality")
    
    @classmethod
    def print_config(cls):
        print("🔧 Current AI Agent Configuration:")
        print(f"   Intent Model: {cls.INTENT_MODEL}")
        print(f"   Code Model: {cls.CODE_MODEL}")
        print(f"   Chat Model: {cls.CHAT_MODEL}")
        print(f"   Timeout: {cls.TIMEOUT}s")
        print(f"   Max Retries: {cls.MAX_RETRIES}")
        print(f"   Auto-fix: {cls.ENABLE_AUTO_FIX}")
        print(f"   Show Code: {cls.SHOW_GENERATED_CODE}")
        print(f"   Safe Mode: {cls.SAFE_MODE}")