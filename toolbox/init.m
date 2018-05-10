global FRAMEWORK_DIR;
FRAMEWORK_DIR = fileparts(mfilename('fullpath'));
LIB_DIR = fullfile(FRAMEWORK_DIR,'lib');%当前目录下创建lib文件

set(0,'defaultAxesFontName', 'Arial');
set(0,'defaultTextFontName', 'Arial');

addpath(genpath(fullfile(FRAMEWORK_DIR)));%添加FRAMEWORK_DIR下所有子目录

% try compile the mex file(s) if needed
bool = exist('SOPD') ~= 0;          
if ~bool
   try
       fprintf('performing first time installation of mex files...');
       wd = cd;
       cd(fullfile(FRAMEWORK_DIR,'helper'));
       mex('SOPD.cpp');
       cd(wd);
       fprintf(' done\n');
   catch ME
       error('failed! You might not be able to run the code.');
   end
end