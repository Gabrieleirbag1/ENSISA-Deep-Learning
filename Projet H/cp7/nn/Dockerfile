# First stage, build the application jar using maven
# pull a container with java 21 and 3.9.9 maven included 
FROM maven:3.9.9-eclipse-temurin-21-alpine AS mvn_builder
RUN mkdir /project
COPY . /project
WORKDIR /project

# package app
RUN --mount=type=cache,id=m2-cache,sharing=shared,target=/root/.m2 \
	mvn clean package -DskipTests

# Second stage, build a custom JRE for the app jar
FROM maven:3.9.9-eclipse-temurin-21-alpine AS jre_builder
RUN mkdir /deployment
WORKDIR /deployment
COPY --from=mvn_builder /project/target/nn-0.0.1-SNAPSHOT.jar nn-0.0.1-SNAPSHOT.jar
#unpack all dependencies for JRE building
RUN jar xf nn-0.0.1-SNAPSHOT.jar

# find requested modules
RUN jdeps --ignore-missing-deps -q \
	--recursive \
	--multi-release 21 \
	--print-module-deps \
	--class-path 'BOOT-INF/lib/*' \
	nn-0.0.1-SNAPSHOT.jar > modules.txt

# Build small JRE image based on the requested modules
RUN jlink \
	--verbose \
	--add-modules $(cat modules.txt) \
	--strip-debug \
	--no-man-pages \
	--no-header-files \
	--compress zip-6 \
	--output optimized-jre-21

# Third stage, put together the custom JRE and the app jar kin a tiny container
FROM alpine:latest
ENV JAVA_HOME=/optimized-jre-21
ENV PATH=$JAVA_HOME/bin:$PATH
# copy JRE from the builder image
COPY --from=jre_builder /deployment/optimized-jre-21 $JAVA_HOME

# copy JAR from the builder image
RUN mkdir /app
RUN addgroup --system web && adduser -S -s /bin/false -G web app web
COPY --from=mvn_builder /project/target/nn-0.0.1-SNAPSHOT.jar /app/app.jar
RUN chown app:web /app/app.jar
USER app

WORKDIR /app
EXPOSE 8080
ENTRYPOINT [ "java", "-jar", "app.jar" ]
