# optype.inspect

A collection of functions for runtime inspection of types, modules, and other
objects.

<table width="415px">
    <tr>
        <th>Function</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>get_args(_)</code></td>
        <td>
            A better alternative to
            <a href="https://docs.python.org/3/library/typing.html#typing.get_args"><code>typing.get_args()</code></a>,
            that
            <ul>
            <li>
                unpacks <code>typing.Annotated</code> and Python 3.12 <code>type _</code> alias
                types (i.e. <code>typing.TypeAliasType</code>),
            </li>
            <li>recursively flattens unions and nested <code>typing.Literal</code> types, and</li>
            <li>raises <code>TypeError</code> if not a type expression.</li>
            </ul>
            Return a <code>tuple[...]</code> of type arguments or parameters.
            <br>
            To illustrate one of the (many) issues with <code>typing.get_args()</code>:
            ```pycon
            >>> from typing import Literal, TypeAlias, get_args
            >>> Falsy: TypeAlias = Literal[None] | Literal[False, 0] | Literal["", b""]
            >>> get_args(Falsy)
            (typing.Literal[None], typing.Literal[False, 0], typing.Literal['', b''])
            ```
            But this is in direct contradiction with the
            <a href="https://typing.python.org/en/latest/spec/literal.html#shortening-unions-of-literals">official typing documentation</a>:
            <blockquote>
            When a <code>Literal</code> is parameterized with more than one value, it’s treated as
            exactly equivalent to the union of those types.
            That is, <code>Literal[v1, v2, v3]</code> is equivalent to
            <code>Literal[v1] | Literal[v2] | Literal[v3]</code>
            </blockquote>
            So this is why <code>optype.inspect.get_args</code> should be used
            ```pycon
            >>> import optype as opt
            >>> opt.inspect.get_args(Falsy)
            (None, False, 0, '', b'')
            ```
            Another issue of <code>typing.get_args()</code> is with Python 3.12
            <code>type _ = ...</code> aliases, which are meant as a replacement for
            <code>_: typing.TypeAlias = ...</code>, and should therefore be treated equally:
            ```pycon
            >>> import typing
            >>> import optype as opt
            >>> type StringLike = str | bytes
            >>> typing.get_args(StringLike)
            ()
            >>> opt.inspect.get_args(StringLike)
            (<class 'str'>, <class 'bytes'>)
            ```
            Clearly, <code>typing.get_args</code> fails miserably here; it would have been better
            if it would have raised an error, but it instead returns an empty tuple,
            hiding the fact that it doesn't support the new <code>type _ = ...</code> aliases.
            But luckily, <code>optype.inspect.get_args</code> doesn't have this problem, and treats
            it just like it treats <code>typing.Alias</code> (and so do the other
            <code>optype.inspect</code> functions).
        </td>
    </tr>
    <tr>
        <td><code>get_protocol_members(_)</code></td>
        <td>
        A better alternative to
        <a href="https://docs.python.org/3/library/typing.html#typing.get_protocol_members"><code>typing.get_protocol_members()</code></a>,
        that
        <ul>
            <li>doesn't require Python 3.13 or above,</li>
            <li>
                supports <a href="https://peps.python.org/pep-0695/">PEP 695</a>
                <code>type _</code> alias types on Python 3.12 and above,
            </li>
            <li>unpacks unions of <code>typing.Literal</code> ...</li>
            <li>... and flattens them if nested within another <code>typing.Literal</code>,</li>
            <li>treats <code>typing.Annotated[T]</code> as <code>T</code>, and</li>
            <li>raises a <code>TypeError</code> if the passed value isn't a type expression.</li>
        </ul>
        Returns a <code>frozenset[str]</code> with member names.
        </td>
    </tr>
    <tr>
        <td><code>get_protocols(_)</code></td>
        <td>
            Returns a <code>frozenset[type]</code> of the public protocols within the
            passed module. Pass <code>private=True</code> to also return the private
            protocols.
        </td>
    </tr>
    <tr>
        <td><code>is_iterable(_)</code></td>
        <td>
            Check whether the object can be iterated over, i.e. if it can be used in a
            <code>for</code> loop, without attempting to do so. If <code>True</code> is
            returned, then the object is an <code>optype.typing.AnyIterable</code>
            instance.
        </td>
    </tr>
    <tr>
        <td><code>is_final(_)</code></td>
        <td>
            Check if the type, method / classmethod / staticmethod / property, is
            decorated with
            <a href="https://docs.python.org/3/library/typing.html#typing.final"><code>@typing.final</code></a>.
            <br>
            Note that a <code>@property</code> won't be recognized unless the
            <code>@final</code> decorator is placed *below* the <code>@property</code>
            decorator. See the function docstring for more information.
        </td>
    </tr>
    <tr>
        <td><code>is_protocol(_)</code></td>
        <td>
            A backport of
            <a href="https://docs.python.org/3/library/typing.html#typing.is_protocol"><code>typing.is_protocol</code></a>
            that was added in Python 3.13, a re-export of
            <a href="https://typing-extensions.readthedocs.io/en/latest/#typing_extensions.is_protocol"><code>typing_extensions.is_protocol</code></a>.
        </td>
    </tr>
    <tr>
        <td><code>is_runtime_protocol(_)</code></td>
        <td>
            Check if the type expression is a <i>runtime-protocol</i>, i.e. a
            <code>typing.Protocol</code> <i>type</i>, decorated with
            <code>@typing.runtime_checkable</code> (also supports
            <code>typing_extensions</code>).
        </td>
    </tr>
    <tr>
        <td><code>is_union_type(_)</code></td>
        <td>
            Check if the type is a
            <a href="https://docs.python.org/3/library/typing.html#typing.Union"><code>typing.Union</code></a>
            type, e.g. <code>str | int</code>.
            <br>
            Unlike <code>isinstance(_, types.Union)</code>, this function also returns
            <code>True</code> for unions of user-defined <code>Generic</code> or
            <code>Protocol</code> types (because those are different union types for
            some reason).
        </td>
    </tr>
    <tr>
        <td><code>is_generic_alias(_)</code></td>
        <td>
            Check if the type is a *subscripted* type, e.g. <code>list[str]</code> or
            <code>optype.CanNext[int]</code>, but not <code>list</code>,
            <code>CanNext</code>.
            <br>
            Unlike <code>isinstance(_, typing.GenericAlias)</code>, this function also
            returns <code>True</code> for user-defined <code>Generic</code> or
            <code>Protocol</code> types (because those are use a different generic alias
            for some reason).
            <br>
            Even though technically <code>T1 | T2</code> is represented as
            <code>typing.Union[T1, T2]</code> (which is a (special) generic alias),
            <code>is_generic_alias</code> will returns <code>False</code> for such union
            types, because calling <code>T1 | T2</code> a subscripted type just doesn't
            make much sense.
        </td>
    </tr>
</table>

> [!NOTE]
> All functions in `optype.inspect` also work for Python 3.12 `type _` aliases
> (i.e. `types.TypeAliasType`) and with `typing.Annotated`.

[UNION]: https://docs.python.org/3/library/typing.html#typing.Union
[LITERAL-DOCS]: https://typing.readthedocs.io/en/latest/spec/literal.html#shortening-unions-of-literals
[@FINAL]: https://docs.python.org/3/library/typing.html#typing.Literal
[GET_ARGS]: https://docs.python.org/3/library/typing.html#typing.get_args
[IS_PROTO]: https://docs.python.org/3.13/library/typing.html#typing.is_protocol
[IS_PROTO_EXT]: https://typing-extensions.readthedocs.io/en/latest/#typing_extensions.is_protocol
[PROTO_MEM]: https://docs.python.org/3.13/library/typing.html#typing.get_protocol_members
