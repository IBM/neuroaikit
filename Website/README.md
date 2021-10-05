## Editing
* Install `jupytext` (e.g. `pip install jupytex`)
* Edit `.ipnb` Jupyter notebooks, which are synced from the respective master `.py` files.
* **Warning**: Always rename files from Jypyter menu rather than renaming `.py` and `.ipynb` files manually.    
* **Warning**: Don't use Autosave in Jypyter. Explicitly CTRL+S after editing in Jupyter, and reload in Jupyter after editing the `.py` files in other editor.

## Publishing

### First time setup
Run in the NeuroAI Toolkit directory:
```
neuroaikit$ git clone -b gh-pages --single-branch https://github.com/IBM/neuroaikit.git ../neuroaikit_gh-pages
```

Install build dependencies:
```
pip install sphinx nbsphinx sphinx-rtd-theme
```

Install PanDoc [https://pandoc.org/installing.html]


### Building the website
Run in the Toolkit directory:
```
neuroaikit$ cd Website
Website$ make clean
Website$ make html
```
Verify locally by opening:
```
Website$ ../../neuroaikit_docs/html/index.html
```

Commit and push: (TODO: not verified!!)
```
cd ../../neuroaikit_gh-pages
neuroaikit_gh-pages$ rm -rf *
neuroaikit_gh-pages$ cp -r ../neuroaikit_docs/html/* . 
neuroaikit_gh-pages$ git add -A && git commit -m "Website update" && git push
```

Check [https://ibm.github.io/neuroaikit/]
Note: Github requires a few minutes to update the website.