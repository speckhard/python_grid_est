from distutils.core import setup
setup(
  name = 'gridtopologyestimation',
  packages = ['estimation', 'examples', 'mutual_information', 'transform_data'],
  version = '0.4',
  description = 'Library to estimate topology of electrical grid using voltage sensor data',
  author = 'Daniel Speckhard',
  author_email = 'dts@stanford.edu',
  url = 'https://github.com/speckhard/Python_Grid_Est', # use the URL to the github repo
  download_url = 'https://github.com/speckhard/Python_Grid_Est/archive/0.4.tar.gz', 
  keywords = ['information-theory', 'smart-grid', 'estimation', 'toplogy', 'mutual-information'], 
  classifiers = [],
  zip_safe = False,
)