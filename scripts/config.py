class Config:
    # Model configuration
    CODE_MODEL = 'deepseek-coder-v2'    
    INTENT_MODEL = "phi3"  
    
    # Execution settings
    TIMEOUT = 300000000
    MAX_RETRIES = 2  # Legacy setting - kept for compatibility
    MAX_FIX_DEPTH = 5  # NEW: Maximum recursive fix attempts
    
    # Memory and safety
    MEMORY_LIMIT_MB = 128
    ENABLE_MEMORY_LIMIT = True
    SAFE_MODE = True
    
    # Auto-fixing settings
    ENABLE_AUTO_FIX = True
    ENABLE_RECURSIVE_FIXING = True  # NEW: Enable recursive error fixing
    
    # Display settings
    SHOW_GENERATED_CODE = True
    SHOW_EXECUTION_STEPS = True
    SHOW_DEBUG_INFO = False
    SHOW_FIX_HISTORY = False  # NEW: Show detailed fix attempt history
    
    # Allowed imports for safety
    ALLOWED_IMPORTS = [
        'math', 'random', 'datetime', 'json', 'csv', 're', 
        'collections', 'itertools', 'functools', 'operator',
        'os', 'sys', 'platform', 'glob', 'pathlib',
        'tkinter', 'pygame', 'PIL', 'numpy', 'pandas', 'matplotlib',
        'requests', 'urllib', 'sqlite3', 'pickle'
    ]
    
    # Model presets
    MODELS = {
        'fast': 'deepseek-r1:1.5b',
        'balanced': 'deepseek-r1:7b',
        'powerful': 'deepseek-r1',
        'large': 'deepseek-r1:32b',
        'massive': 'deepseek-r1:671b',
        'coder': 'deepseek-coder-v2',
        'reasoner': 'deepseek-r1'
    }
    
    @classmethod
    def use_model_preset(cls, preset: str):
        """Apply a model preset to all model types"""
        if preset in cls.MODELS:
            model = cls.MODELS[preset]
            cls.INTENT_MODEL = model
            cls.CODE_MODEL = model
            print(f"🔄 Switched to {preset} model: {model}")
        else:
            print(f"❌ Unknown preset: {preset}")
            print(f"Available presets: {list(cls.MODELS.keys())}")
    
    @classmethod
    def set_performance_mode(cls, mode: str):
        """Set performance mode with optimized settings for recursive fixing"""
        if mode == "fast":
            cls.TIMEOUT = 15
            cls.MAX_FIX_DEPTH = 2
            cls.SHOW_GENERATED_CODE = False
            cls.SHOW_FIX_HISTORY = False
            cls.use_model_preset("fast")
        elif mode == "balanced":
            cls.TIMEOUT = 30
            cls.MAX_FIX_DEPTH = 5
            cls.SHOW_GENERATED_CODE = True
            cls.SHOW_FIX_HISTORY = False
            cls.use_model_preset("balanced")
        elif mode == "quality":
            cls.TIMEOUT = 60
            cls.MAX_FIX_DEPTH = 8
            cls.SHOW_GENERATED_CODE = True
            cls.SHOW_FIX_HISTORY = True
            cls.use_model_preset("powerful")
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: fast, balanced, quality")
    
    @classmethod
    def set_recursive_mode(cls, aggressive: bool = True):
        """Configure recursive fixing aggressiveness"""
        if aggressive:
            cls.MAX_FIX_DEPTH = 10
            cls.ENABLE_RECURSIVE_FIXING = True
            cls.SHOW_FIX_HISTORY = True
            print("🔧 Enabled aggressive recursive fixing (depth: 10)")
        else:
            cls.MAX_FIX_DEPTH = 3
            cls.ENABLE_RECURSIVE_FIXING = True
            cls.SHOW_FIX_HISTORY = False
            print("🔧 Enabled conservative recursive fixing (depth: 3)")
    
    @classmethod
    def disable_recursive_fixing(cls):
        """Disable recursive fixing (fall back to simple retry)"""
        cls.ENABLE_RECURSIVE_FIXING = False
        cls.MAX_FIX_DEPTH = 1
        print("🔧 Disabled recursive fixing - using simple retry only")
    
    @classmethod
    def print_config(cls):
        """Display current configuration"""
        print("🔧 Current AI Agent Configuration:")
        print(f"   Intent Model: {cls.INTENT_MODEL}")
        print(f"   Code Model: {cls.CODE_MODEL}")
        print(f"   Timeout: {cls.TIMEOUT}s")
        print(f"   Recursive Fixing: {'✅ Enabled' if cls.ENABLE_RECURSIVE_FIXING else '❌ Disabled'}")
        print(f"   Max Fix Depth: {cls.MAX_FIX_DEPTH}")
        print(f"   Show Generated Code: {cls.SHOW_GENERATED_CODE}")
        print(f"   Show Fix History: {cls.SHOW_FIX_HISTORY}")
        print(f"   Safe Mode: {cls.SAFE_MODE}")
        print(f"   Memory Limit: {cls.MEMORY_LIMIT_MB}MB")