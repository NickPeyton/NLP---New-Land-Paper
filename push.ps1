"PUSHING ALL YOUR NONSENSE TO HUGGINGFACE AND GITHUB"
"---------------------------------------------------"
cd ./Code/ml_models
git add .
git commit -m (Get-Date).tostring()
git push
"---------------------------------------------------"
cd ../..
git add .
git commit -m (Get-Date).tostring()
git push
"---------------------------------------------------"
Read-Host -prompt "Press [Enter] to exit"