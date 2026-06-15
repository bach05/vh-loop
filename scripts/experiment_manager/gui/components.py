from typing import Type, Any, Dict, get_origin, get_args, Union, Literal
from pydantic import BaseModel
from nicegui import ui
import json

def render_pydantic_form(model_class: Type[BaseModel], data: Dict[str, Any], on_change: Any = None) -> Dict[str, Any]:
    """
    Renders NiceGUI input elements matching the fields of a Pydantic model class.
    Updates and returns a dict with the form values.
    Supports nested models, lists of models, and dictionary values.
    """
    # Initialize dictionary structure for nested/nested values if not present
    for name, field in model_class.model_fields.items():
        if name not in data:
            annotation = field.annotation
            origin = get_origin(annotation)
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                data[name] = {}
            elif origin is list:
                data[name] = []
            elif origin is dict or annotation is dict:
                data[name] = {}
            else:
                data[name] = field.default if field.default is not None else None

    # Render inputs
    for name, field in model_class.model_fields.items():
        label = name.replace("_", " ").title()
        annotation = field.annotation
        
        # Handle Union/Optional types
        origin = get_origin(annotation)
        if origin is Union:
            args = get_args(annotation)
            # Filter out NoneType to get the main type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                annotation = non_none_args[0]
                origin = get_origin(annotation)
            else:
                # Fallback to first non-none arg for simplified forms
                annotation = non_none_args[0]
                origin = get_origin(annotation)

        # Check if Dict
        if origin is dict or annotation is dict:
            if data[name] is None:
                data[name] = {}
            initial_val = json.dumps(data[name], indent=2)
            
            def make_change_handler(n=name, d=data):
                def handler(e):
                    try:
                        d[n] = json.loads(e.value)
                    except json.JSONDecodeError:
                        pass
                return handler

            with ui.row().classes('w-full items-start justify-between my-2'):
                ui.label(label).classes('text-sm font-medium w-1/3 text-slate-600 pt-2')
                ui.textarea(value=initial_val, on_change=make_change_handler()).classes('w-2/3').props('autogrow outlined dense')
            continue

        # Check if List of BaseModel
        if origin is list:
            args = get_args(annotation)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                item_class = args[0]
                if data[name] is None:
                    data[name] = []
                
                with ui.card().classes('w-full q-pa-md my-2 bg-slate-50/50 border border-dashed'):
                    with ui.row().classes('w-full justify-between items-center border-b pb-2 mb-2'):
                        ui.label(label).classes('text-lg font-semibold text-slate-700')
                        
                        def add_item(lst=data[name], cls=item_class):
                            default_item = {}
                            for sub_name, sub_field in cls.model_fields.items():
                                default_item[sub_name] = sub_field.default if sub_field.default is not None else None
                            lst.append(default_item)
                            if on_change:
                                on_change()
                                
                        ui.button("Add Item", on_click=add_item).classes('bg-indigo-600 text-white text-xs')
                    
                    for idx, item in enumerate(data[name]):
                        with ui.card().classes('w-full q-pa-sm my-1 border relative bg-white'):
                            with ui.row().classes('w-full justify-between items-center border-b pb-1 mb-2'):
                                ui.label(f"Item #{idx + 1}").classes('font-medium text-sm text-slate-600')
                                
                                def remove_item(i=idx, lst=data[name]):
                                    lst.pop(i)
                                    if on_change:
                                        on_change()
                                        
                                ui.button(icon="delete", on_click=remove_item).props('flat round dense color=red')
                            render_pydantic_form(item_class, item, on_change)
                continue

        # Check if nested BaseModel
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            with ui.card().classes('w-full q-pa-md my-2 bg-white shadow-sm'):
                ui.label(label).classes('text-lg font-semibold border-b pb-1 w-full text-slate-700')
                if data[name] is None:
                    data[name] = {}
                render_pydantic_form(annotation, data[name], on_change)
        else:
            # Render simple types
            with ui.row().classes('w-full items-center justify-between my-1'):
                ui.label(label).classes('text-sm font-medium w-1/3 text-slate-600')
                
                # Check for Literal types (dropdown options)
                is_literal = False
                if str(annotation).startswith("typing.Literal") or "Literal" in str(annotation):
                    # Extract literal values
                    import re
                    choices = re.findall(r"'(.*?)'", str(annotation))
                    if not choices:
                        choices = re.findall(r'"(.*?)"', str(annotation))
                    if choices:
                        ui.select(choices, label=label).bind_value(data, name).classes('w-2/3')
                        is_literal = True
                
                if not is_literal:
                    if annotation is bool:
                        ui.switch().bind_value(data, name)
                    elif annotation is int:
                        ui.number(format='%.0f').bind_value(data, name).classes('w-2/3')
                    elif annotation is float:
                        ui.number(format='%.4f').bind_value(data, name).classes('w-2/3')
                    else:
                        ui.input().bind_value(data, name).classes('w-2/3')

    return data
