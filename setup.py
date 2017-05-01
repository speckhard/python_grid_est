from distutils.core import setup
from distutils import util
if __name__ == "__main__":
    pathMySubPackage1 = util.convert_path('discretetopology/estimation')
    pathMySubPackage2 = util.convert_path('discretetopology/mutual_information')
    pathMySubPackage3 = util.convert_path('discretetopology/transform_data')
    setup(
        name='discretetopology',
        package_dir = {
            'discretetopology': 'discretetopology',
            'discretetopology.estimation': pathMySubPackage1,
            'discretetopology.mutual_infromation': pathMySubPackage2,
            'discretetopology.tranform_data': pathMySubPackage3},
        packages=['discretetopology', 'discretetopology.estimation',
                  'discretetopology.mutual_infromation',
                  'discretetopology.tranform_data'],
        version = '0.9',
        description = 'Library to estimate topology of electrical grid using voltage sensor data',
        author = 'Daniel Speckhard',
        author_email = 'dts@stanford.edu',
        url = 'https://github.com/speckhard/Python_Grid_Est', # use the URL to the github repo
        download_url = 'https://github.com/speckhard/Python_Grid_Est/archive/0.9.tar.gz', 
        keywords = ['information-theory', 'smart-grid', 'estimation', 'toplogy', 'mutual-information'], 
        classifiers = [],
        zip_safe = False,
      )