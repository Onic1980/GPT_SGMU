# Saprykin Dmitry
# Palliative chat bot

## Installing

If not already in a virtual environement, create and use one.
Read about it in the Python documentation: [venv — Creation of virtual environments](https://docs.python.org/3/library/venv.html).

```
python -m venv .venv
```

Linux
```
source .venv/Scripts/activate
```

Mac OS
```
source .venv/bin/activate
```

Windows
```
.venv/Scripts/activate.bat
```

Install the dependencies:

```
pip install -r requirements.txt
```

## Create index base knowledge

```
bash ./scripts/create_index.sh
```

## Run backend

Start

```
some command
```

## Testing

Execute tests from the library's folder (after having loaded the virtual environment,
see above) as follows:

```
python -m pytest tests/
