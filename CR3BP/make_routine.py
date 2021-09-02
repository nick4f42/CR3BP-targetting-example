import sympy as sp
from sympy.utilities.codegen import (
    Routine, InputArgument, OutputArgument, Result)
from sympy.codegen.rewriting import (
    create_expand_pow_optimization, optimize, optims_c99)

_expand_opt = create_expand_pow_optimization(3, base_req=lambda b: not b.is_Function)

def make_routine(name, in_symbols, out_eqs=(), result=None, global_vars=(),
                 var_map=None, cse=True, cse_kwargs=None, optims=optims_c99):
    """Return a sympy codegen routine with optimizations.

    Args:
        name: Routine name.
        in_symbols: List of input symbols.
        out_eqs (optional): List of `sympy.Equality` like `out_sym = expr`.
            The left-hand side is an output (Matrix)Symbol and the right-hand
            side is the expression that be assigned to it.
        result (optional): The symbol to return.
        global_vars (optional): Global variables expected by the codegen.
        var_map (optional): Map to these symbols for optimizations and unmap
            for the final result. Useful for dealing with MatrixSymbols to make
            optimizations work correctly.
        cse (optional): Whether to use cse. Default is true.
        cse_kwargs (optional): Dictionary of kwargs to `sympy.cse`.
        optims (optional): Optimizations to pass to sympy's `optimize`.
            Default is the C99 standard optimizations. See for more details:
            https://docs.sympy.org/latest/modules/codegen.html#sympy.codegen.rewriting.optimize
    
    Returns:
        A sympy.utilities.codegen.Routine object that may be used with a CodeGen
        object.
    """
    if var_map is None:
        var_map = {}
    var_rmap = {j:i for i,j in var_map.items()}

    if cse_kwargs is None:
        cse_kwargs = {}
    
    in_args = []
    for s in in_symbols:
        dims = ([(0, dim - 1) for dim in s.shape]
            if isinstance(s, sp.MatrixSymbol) else None)
        in_args.append(InputArgument(s, dimensions=dims))
    
    out_vars = [eq.lhs for eq in out_eqs]
    
    calc_exprs = [eq.rhs.xreplace(var_map) for eq in out_eqs]

    opt = lambda expr: _expand_opt(optimize(expr, optims))
    
    if cse:
        if result is not None:
            calc_exprs.append(result.xreplace(var_map))
            common, (*out_exprs, result) = sp.cse(calc_exprs, **cse_kwargs)
        else:
            common, out_exprs = sp.cse(calc_exprs, **cse_kwargs)
        
        local_vars = [Result(opt(expr.xreplace(var_rmap)), var, var)
                        for var, expr in common]
    else:
        out_exprs = calc_exprs
        local_vars = ()
    
    out_args = [OutputArgument(var, var, opt(expr.xreplace(var_rmap)))
                for var, expr in zip(out_vars, out_exprs)]
    
    results = [] if result is None else [Result(result)]
    return Routine(name, in_args + out_args, results, local_vars, global_vars)
