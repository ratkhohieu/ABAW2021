APP_NAME=svenclary/pytorch_cv:lastest
CONTAINER_NAME=ABAW

run: ## Run container
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME} \
		-v /mnt/DATA1/hung/ABAW:/workspace \
		-v /mnt/DATA1/emotion_dataset/AffectNet/Manually_Annotated_compressed/Manually_Annotated_Images:/data_emotion \
		-v $(shell pwd) $(APP_NAME) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it ${CONTAINER_NAME} bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}; docker rm ${CONTAINER_NAME}