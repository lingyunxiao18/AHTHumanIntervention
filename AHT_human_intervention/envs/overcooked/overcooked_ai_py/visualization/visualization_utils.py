try:
    from IPython.display import Image, display
    from ipywidgets import IntSlider, interactive
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    # Create dummy functions for when IPython is not available
    def Image(*args, **kwargs):
        return None
    
    def display(*args, **kwargs):
        pass
    
    def IntSlider(*args, **kwargs):
        return None
    
    def interactive(*args, **kwargs):
        return None


def show_image_in_ipython(data, *args, **kwargs):
    if IPYTHON_AVAILABLE:
        display(Image(data, *args, **kwargs))
    else:
        print("IPython not available - skipping image display")


def ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    if not IPYTHON_AVAILABLE:
        print("IPython not available - skipping slider")
        return None
    
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))

    return interactive(display_f, **{slider_label: IntSlider(min=0, max=len(image_pathes_list) - 1, step=1)})


def show_ipython_images_slider(image_pathes_list, slider_label="", first_arg=0):
    if not IPYTHON_AVAILABLE:
        print("IPython not available - skipping slider")
        return None
    
    def display_f(**kwargs):
        display(Image(image_pathes_list[kwargs[slider_label]]))

    display(interactive(display_f, **{slider_label: IntSlider(min=0, max=len(image_pathes_list) - 1, step=1)}))
