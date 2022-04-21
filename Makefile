include .env

heroku-login:
	heroku container:login

build: heroku-login
	docker build -t registry.heroku.com/${NAME}/web .

push: build
	docker push registry.heroku.com/${NAME}/web

release: push
	heroku container:release web --app ${NAME}

.PHONY: heroku-login build push