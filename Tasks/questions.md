### Answer the following questions in you own words.

> It's not necessary that you havee to know and answer all the questions. Just answer the ones
> you know and write in your own words.

1. Give the difference between the remotes - upstream and origin - with an example.

You answer:
-Origin is your repository where you push your changes 
-Upstream is the main repository 

2. You have two branches A and B and you have currently made some changes in branch A.
You want to move into branch B but do not want to commit the current changes in branch A.
What will you do?

You answer:
We use git stash to save temporary changes.
Like if we have to go from Branch-A to Branch-B without committing , we will do -
```bash
Git stash 
Git checkout B
```


3. You were assigned a work to implement a feature and create a PR to your organization's remote repository.
For this you made a branch (say A) and made some changes and commited them. Now you moved to some other branch 
(say B) to do some other assigned work. But later you realisd that have to complete the task assigned earlier 
first and commited some changes in branch B which are meant for branch A. How will you use git to bring the 
changes from branch B to branch A?

You answer:

3. What is the difference between fetching changes and pulling changes?

Your answer:
->Git fetch downloads changes from remote but doesn't apply them. It keeps your local branch unchanged.
->Git pull is fetch + merge. It downloads changes and applies them immediately.
->Git fetch allows us to review the changes before applying them , it downloads the commits first whereas git pull applies the changes right away. 

4. What does -i flag stand for? What is it's significance in git?

You answer:
It means interactive mode. It lets us edit commits. 

5. You are working in an organization that follows very strict guidelines for PRs and commits.
You made three commits in your PR and the maintainer says you were supposed to make a single commit.
What will you do in this case?

You answer:

6. Explain `git merge` and `git rebase` with example(s).

You answer:
git merge: Combines two branches, creates a new merge commit. This keeps history therefore a little messy. 
```bash
git checkout main
git merge feature
```
git rebase This cleans history but rewrites commit. 
```bash
git checkout feature
git rebase main
```

7. Write the flow how you create a repository and push changes to it. Also mention the commands used at each step.

You answer:
1. Initialize repo 
```bash
git init
```
2. Add files 
```bash
git add .
```
3. Commit changes 
```bash
git commit -m "first commit"
```
4. Add remote 
```bash
git remote add origin
```
5. push
```bash
git push -u origin main
```
-u flag stands for --set-upstream. 
If we use -u then afterwards we can just do 
```bash
git push
git pull
```
or else we will have to do this everytime - 
```bash
git push origin main
git pull origin main
```

8. How would you prevent a file or folder from getting tracked by git?

Your answer:
Add file/folder name to .gitignore.

9. You did not implement the step you mentioned in question 8 and now you have committed and pushed your database's
secret key to the github. How will you remove the key from your git's commit history to avoid any misuse?

You answer:
remove the commit and do a hard reset. 

---