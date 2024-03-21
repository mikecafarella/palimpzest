# Palimpzest (PZ)
Palimpzest is a framework for writing document-centric programs. It will help you marshal, clean, extract, transform, and integrate documents and data. The LLM compute platform is going to read and write a lot of documents; Palimpzest is how programmers can control it.

Some nice things Palimpzest does for you:
- Write ETL-style programs very quickly. The code is reusable, composable, and shareable.
- Declarative data quality management: you focus on the app, and let the system figure out quality improvements. Don't bother with model details; let the compiler handle it.
- Declarative runtime platform management: you tell the system how much money you want to spend, and let the system figure out how to make the program as fast as possible.
- Automatic data marshaling. Data naming, sampling, and caching are first-class concepts
- Ancillary material comes for free: annotation tools, accuracy reports, data version updates, useful provenance records, etc

Some target use cases for Palimpzest:
- **Information Extraction**: Extract a useable pandemic model from a scientific paper that is accompanied by its code and test datasets
- **Scientific Discovery**: Extract all the data tuples from every experiment in every battery electrolyte paper ever written, then write a simple query on them 
- **Data Integration**: Integrate multimodal bioinformatics data and make a nice exploration tool
- **Document Processing**: Process all the footnotes in all the bank regulatory statements to find out which ones are in trouble
- **Data Mining (get it???)**: Comb through historical maps to find likely deposits of critical minerals
- **Digital Twins**: Create a system to understand your software team's work. Integrate GitHub commits, bug reports, and the next release's feature list into a single integrated view. Then add alerting, summaries, rankers, explorers, etc.
- **Next-Gen Dashboards**: Integrate your datacenter's logs with background documentation, then ask for hypotheses about a bug you're seeing in Datadog. Go beyond the ocean of 2d plots.

# Getting started
You can install the Palimpzest package and CLI on your machine by cloning this repository and running:
```bash
$ git clone git@github.com:mikecafarella/palimpzest.git
$ cd palimpzest
$ pip install .
```


## Palimpzest CLI
Installing Palimpzest also installs its CLI tool `pz` which provides users with basic utilities for creating and managing their own Palimpzest system. Running `pz --help` diplays an overview of the CLI's commands:
```bash
$ pz --help
Usage: pz [OPTIONS] COMMAND [ARGS]...

  The CLI tool for Palimpzest.

Options:
  --help  Show this message and exit.

Commands:
  help (h)                        Print the help message for PZ.
  init (i)                        Initialize data directory for PZ.
  ls-data (ls,lsdata)             Print a table listing the datasets
                                  registered with PZ.
  register-data (r,reg,register)  Register a data file or data directory with
                                  PZ.
  rm-data (rm,rmdata)             Remove a dataset that was registered with
                                  PZ.
```

Users can initialize their own system by running `pz init`. This will create Palimpzest's working directory in `~/.palimpzest`:
```bash
$ pz init
Palimpzest system initialized in: /Users/matthewrusso/.palimpzest
```

If we list the set of datasets registered with Palimpzest, we'll see there currently are none:
```bash
$ pz ls
+------+------+------+
| Name | Type | Path |
+------+------+------+
+------+------+------+

Total datasets: 0
```

### Registering Datasets
To add (or "register") a dataset with Palimpzest, we can use the `pz register-data` command (also aliased as `pz reg`) to specify that a file or directory at a given `--path` should be registered as a dataset with the specified `--name`:
```bash
$ pz reg --path README.md --name rdme
Registered rdme
```

If we list Palimpzest's datasets again we will see that `README.md` has been registered under the dataset named `rdme`:
```bash
$ pz ls
+------+------+------------------------------------------+
| Name | Type |                   Path                   |
+------+------+------------------------------------------+
| rdme | file | /Users/matthewrusso/palimpzest/README.md |
+------+------+------------------------------------------+

Total datasets: 1
```

To remove a dataset from Palimpzest, simply use the `pz rm-data` command (also aliased as `pz rm`) and specify the `--name` of the dataset you would like to remove:
```bash
$ pz rm --name rdme
Deleted rdme
```

Finally, listing our datasets once more will show that the dataset has been deleted:
```bash
$ pz ls
+------+------+------+
| Name | Type | Path |
+------+------+------+
+------+------+------+

Total datasets: 0
```

### Cache Management
Palimpzest will cache intermediate results by default. It can be useful to remove them from the cache when trying to evaluate the performance improvement(s) of code changes. We provide a utility command `pz clear-cache` (also aliased as `pz clr`) to clear the cache:
```bash
$ pz clr
Cache cleared
```

### Config Management
You may wish to work with multiple configurations of Palimpzest in order to, e.g., evaluate the difference in performance between various LLM services for your data extraction task. To see the config Palimpzest is currently using, you can run the `pz print-config` command (also aliased as `pz config`):
```bash
$ pz config
--- default ---
filecachedir: /some/local/filepath
llmservice: openai
name: default
parallel: false
```
By default, Palimpzest uses the configuration named `default`. As shown above, if you run a script using Palimpzest out-of-the-box, it will use OpenAI endpoints for all of its API calls.

Now, let's say you wanted to try using [together.ai's](https://www.together.ai/) for your API calls, you could do this by creating a new config with the `pz create-config` command (also aliased as `pz cc`):
```bash
$ pz cc --name together-conf --llmservice together --parallel True --set
Created and set config: together-conf
```
The `--name` parameter is required and specifies the unique name for your config. The `--llmservice` and `--parallel` options specify the service to use and whether or not to process files in parallel. Finally, if the `--set` flag is present, Palimpzest will update its current config to point to the newly created config.

We can confirm that Palimpzest checked out our new config by running `pz config`:
```bash
$ pz config
--- together-conf ---
filecachedir: /some/local/filepath
llmservice: together
name: together-conf
parallel: true
```

You can switch which config you are using at any time by using the `pz set-config` command (also aliased as `pz set`):
```bash
$ pz set --name default
Set config: default

$ pz config
--- default ---
filecachedir: /some/local/filepath
llmservice: openai
name: default
parallel: false

$ pz set --name together-conf
Set config: together-conf

$ pz config
--- together-conf ---
filecachedir: /some/local/filepath
llmservice: together
name: together-conf
parallel: true
```

Finally, you can delete a config with the `pz rm-config` command (also aliased as `pz rmc`):
```bash
$ pz rmc --name together-conf
Deleted config: together-conf
```
Note that you cannot delete the `default` config, and if you delete the config that you currently have set, Palimpzest will set the current config to be `default`.

## Configuring for Parallel Execution

There are a few things you need to do in order to use remote parallel services.

If you want to use parallel LLM execution on together.ai, you have to modify the config.yaml (by default, Palimpzest uses `~/.palimpzest/config_default.yaml`) so that `llmservice: together` and `parallel: true` are set.

If you want to use parallel PDF processing at modal.com, you have to:
1. Set `pdfprocessing: modal` in the config.yaml file.
2. Run `modal deploy src/palimpzest/tools/allenpdf.py`.  This will remotely install the modal function so you can run it. (Actually, it's probably already installed there, but do this just in case.  Also do it if there's been a change to the server-side function inside that file.)

## Configuring for Code Generation Solution

**Currently code generation is inconsistent with parallel execution.** The code snippets are generated once receiving the first input data, and LLM-annotated validation examples are accumulated during execution. Thus parallelism in code generation does not work.

If you want to enable LLM generating code to perform batch operations, you have to modify the config.yaml (by default, Palimpzest uses `~/.palimpzest/config_default.yaml`) so that `codegen: true` and `parallel: false` are set.

Furthermore, you can modify the following parameters to influence the code generation process:
- `codegen_num_ensemble`: how many parallel code snippets to generate. Default to `4`.
- `codegen_validation`: whether to fix code based on error feedback. Default to `false`. (TODO, no actual influence now)
- `codegen_num_iterations`: number of max iterations for code fixing. Must have `codegen_validation: true` to have any effect. Default to `5`.
- `codegen_num_max_examples`: number of max validation examples for code fixing. Must have `codegen_validation: true` to have any effect. Default to `20`.
- `codegen_logging`: whether to print codegen information. This also changes the code output: for logging purpose, if enabled, all codegen returned terms are labelled as `(code extracted)`. Default to `false`.

## Python Demo

Below are simple instructions to run pz on a test data set of enron emails that is included with the system:

- Initialize the configuration by running `pz --init`.

- Add the enron data set with:
`pz reg --path testdata/enron-tiny --name enron-tiny`
then run it through the test program with:
      `tests/simpleDemo.py --task enron --datasetid enron-tiny`

- Add the test paper set with:
    `pz reg --path testdata/pdfs-tiny --name pdfs-tiny`
then run it through the test program with:
`tests/simpleDemo.py --task paper --datasetid pdfs-tiny`


- Palimpzest defaults to using OpenAI. You’ll need to export an environment variable `OPENAI_API_KEY`


