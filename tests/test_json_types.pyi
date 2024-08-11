import optype as opt

def f_val(obj: opt.json.Value, /) -> str: ...
def f_aval(obj: opt.json.AnyValue, /) -> str: ...

val: opt.json.Value
val_arr_obj: opt.json.Value | opt.json.Array | opt.json.Object
aval: opt.json.AnyValue
aval_aarr_aobj: opt.json.AnyValue | opt.json.AnyArray | opt.json.AnyObject
aval_val: opt.json.AnyValue | opt.json.Value

_ = f_val(val_arr_obj)

_ = f_aval(val)
_ = f_aval(aval_aarr_aobj)
_ = f_aval(aval_val)
