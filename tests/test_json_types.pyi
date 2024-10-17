import optype as o

def f_val(obj: o.json.Value, /) -> str: ...
def f_aval(obj: o.json.AnyValue, /) -> str: ...

val: o.json.Value
val_arr_obj: o.json.Value | o.json.Array | o.json.Object
aval: o.json.AnyValue
aval_aarr_aobj: o.json.AnyValue | o.json.AnyArray | o.json.AnyObject
aval_val: o.json.AnyValue | o.json.Value

_ = f_val(val_arr_obj)

_ = f_aval(val)
_ = f_aval(aval_aarr_aobj)
_ = f_aval(aval_val)
